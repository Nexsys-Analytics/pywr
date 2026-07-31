"""Pywr-side optimisation adapter for Borg backends.

This module deliberately contains no dependency on ``pywr-borg``.  It exposes
Pywr's model, decision variables and raw recorder values in the same adapter
layer used by the Platypus, PyGMO and Pymoo integrations.  Borg-specific
semantics (epsilons, objective transformation, constraint violations, native
execution and MPI) belong to the external ``pywr-borg`` backend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import (
    BaseOptimisationWrapper,
    ModelCache,
    cache_constraints,
    cache_objectives,
    cache_variable_parameters,
)


@dataclass(frozen=True, slots=True)
class BorgEvaluation:
    """Raw values produced by one Pywr model evaluation.

    The values intentionally retain Pywr semantics.  In particular, objective
    directions and constraint violations are not transformed here; those are
    backend concerns handled by ``pywr-borg``.
    """

    objectives: tuple[float, ...]
    constraints: tuple[float, ...]
    metrics: tuple[float, ...]


class BorgWrapper(BaseOptimisationWrapper):
    """Expose a Pywr model to a Borg backend without importing Borg itself.

    ``source`` may be either a normal Pywr model source accepted by
    :class:`BaseOptimisationWrapper` or an already-loaded model instance.  The
    latter keeps the existing ``pywr-borg`` public API usable while moving the
    model-facing responsibility into ``pywr.optimisation``.
    """

    def __init__(self, source: Any, *args: Any, **kwargs: Any) -> None:
        self._provided_model = None
        self._provided_cache = None

        if _looks_like_model(source):
            self._provided_model = source
            source = None

        super().__init__(source, *args, **kwargs)

        # Force the standard optimisation caches to be resolved at construction
        # time so malformed models fail before an optimisation starts.
        if not self.model_variables:
            raise ValueError("At least one variable must be defined.")
        if not self.model_objectives:
            raise ValueError("At least one objective must be defined.")

    @property
    def _cached(self):
        if self._provided_model is None:
            return super()._cached

        if self._provided_cache is None:
            model = self._provided_model
            if getattr(model, "dirty", True):
                model.setup()

            cache = ModelCache()
            cache.model = model
            cache.variables, cache.variable_map = cache_variable_parameters(model)
            cache.objectives = cache_objectives(model)
            cache.constraints = cache_constraints(model)
            self._provided_cache = cache

        return self._provided_cache

    def set_variables(self, values: Sequence[float]) -> None:
        """Apply one flattened decision vector to the wrapped Pywr model."""

        expected = self.model_variable_map[-1]
        if len(values) != expected:
            raise ValueError(f"got {len(values)} decision values; expected {expected}")

        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("decision vector contains NaN or infinity")

        for ivar, variable in enumerate(self.model_variables):
            section = slice(self.model_variable_map[ivar], self.model_variable_map[ivar + 1])
            value_slice = array[section]
            offset = 0

            if variable.double_size:
                stop = variable.double_size
                variable.set_double_variables(
                    np.asarray(value_slice[offset:stop], dtype=np.float64)
                )
                offset = stop

            if variable.integer_size:
                stop = offset + variable.integer_size
                integers = np.rint(value_slice[offset:stop]).astype(np.int32)
                variable.set_integer_variables(integers)
                offset = stop

            if offset != len(value_slice):
                raise ValueError(
                    f'variable "{variable.name}" did not consume its decision-vector slice'
                )

    def evaluate(
        self,
        values: Sequence[float],
        *,
        metric_recorders: Sequence[Any] = (),
    ) -> BorgEvaluation:
        """Run Pywr once and return untransformed recorder aggregates."""

        self.set_variables(values)
        self.run_stats = self.model.run()

        objectives = tuple(float(item.aggregated_value()) for item in self.model_objectives)
        constraints = tuple(float(item.aggregated_value()) for item in self.model_constraints)
        metrics = tuple(float(item.aggregated_value()) for item in metric_recorders)

        values_to_check = (*objectives, *constraints, *metrics)
        if any(not np.isfinite(value) for value in values_to_check):
            raise ValueError("Pywr evaluation returned NaN or infinity")

        return BorgEvaluation(
            objectives=objectives,
            constraints=constraints,
            metrics=metrics,
        )


def _looks_like_model(value: Any) -> bool:
    return all(
        hasattr(value, attribute)
        for attribute in ("run", "variables", "objectives", "constraints")
    )
