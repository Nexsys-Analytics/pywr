from types import SimpleNamespace

import numpy as np
import pytest

from pywr.optimisation.borg import BorgWrapper


class _Variable:
    name = "decision"
    double_size = 1
    integer_size = 1

    def __init__(self):
        self.doubles = np.array([0.0])
        self.integers = np.array([0], dtype=np.int32)

    def get_double_lower_bounds(self):
        return np.array([-10.0])

    def get_double_upper_bounds(self):
        return np.array([10.0])

    def get_integer_lower_bounds(self):
        return np.array([0], dtype=np.int32)

    def get_integer_upper_bounds(self):
        return np.array([10], dtype=np.int32)

    def set_double_variables(self, values):
        self.doubles = np.asarray(values, dtype=np.float64)

    def set_integer_variables(self, values):
        self.integers = np.asarray(values, dtype=np.int32)


class _Recorder:
    def __init__(self, name, getter):
        self.name = name
        self._getter = getter

    def aggregated_value(self):
        return self._getter()


class _Model:
    def __init__(self):
        self.dirty = True
        self.variable = _Variable()
        self.variables = [self.variable]
        self.objectives = [_Recorder("objective", self._objective)]
        self.constraints = [_Recorder("constraint", self._constraint)]
        self.metric = _Recorder("metric", self._metric)
        self.recorders = [*self.objectives, *self.constraints, self.metric]
        self.setup_calls = 0
        self.run_calls = 0

    def setup(self):
        self.setup_calls += 1
        self.dirty = False

    def run(self):
        self.run_calls += 1
        return SimpleNamespace(time_taken=0.0, speed=0.0)

    def _objective(self):
        return float(self.variable.doubles[0] + self.variable.integers[0])

    def _constraint(self):
        return float(self.variable.doubles[0] - self.variable.integers[0])

    def _metric(self):
        return float(self.variable.doubles[0] * self.variable.integers[0])


def test_borg_wrapper_evaluates_raw_pywr_values():
    model = _Model()
    wrapper = BorgWrapper(model)

    result = wrapper.evaluate([2.5, 3.0], metric_recorders=[model.metric])

    assert model.setup_calls == 1
    assert model.run_calls == 1
    assert model.variable.doubles.tolist() == [2.5]
    assert model.variable.integers.tolist() == [3]
    assert result.objectives == (5.5,)
    assert result.constraints == (-0.5,)
    assert result.metrics == (7.5,)


def test_borg_wrapper_rounds_integer_decision_values():
    model = _Model()
    wrapper = BorgWrapper(model)

    wrapper.set_variables([1.25, 3.6])

    assert model.variable.doubles.tolist() == [1.25]
    assert model.variable.integers.tolist() == [4]


def test_borg_wrapper_rejects_invalid_decision_vectors():
    wrapper = BorgWrapper(_Model())

    with pytest.raises(ValueError, match="expected 2"):
        wrapper.set_variables([1.0])

    with pytest.raises(ValueError, match="NaN or infinity"):
        wrapper.set_variables([1.0, np.nan])
