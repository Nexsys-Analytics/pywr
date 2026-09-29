import subprocess
import sys
import textwrap

import pytest

from pywr.solvers import solver_registry

GLPK_SOLVER_NAMES = ["glpk", "glpk-edge"]

# Generous bound on a child that builds and solves two two-node models; only a hang exceeds it.
CHILD_TIMEOUT_SECONDS = 120

# The child runs in its own process because the failure mode being guarded against is GLPK aborting the interpreter.
CHILD_SCRIPT = textwrap.dedent("""
    import gc
    import threading

    from pywr.core import Model
    from pywr.nodes import Input, Output

    SOLVER = {solver!r}

    gc.disable()


    def build_and_run_cyclic_model():
        model = Model(solver=SOLVER)
        Output(model, "demand", max_flow=5.0, cost=-10.0)
        Input(model, "supply", max_flow=10.0).connect(model.nodes["demand"])
        model.run()
        # Only the cyclic collector can free the model now.
        model.self_reference = model


    build_and_run_cyclic_model()

    collector = threading.Thread(target=gc.collect)
    collector.start()
    collector.join()

    # The creating thread frees whatever the collector thread queued, then keeps solving.
    build_and_run_cyclic_model()
    gc.collect()
    print("done")
    """)


@pytest.mark.parametrize("solver_name", GLPK_SOLVER_NAMES)
def test_solver_freed_by_another_thread_does_not_abort(solver_name):
    if solver_name not in [s.name for s in solver_registry]:
        pytest.skip(f"Solver {solver_name!r} is not available.")

    result = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT.format(solver=solver_name)],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "done"
