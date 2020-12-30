"""Collection of tests for :mod:`orion.plotting.orion.plotting.PlotAccessor`."""
import pytest

from orion.plotting.base import PlotAccessor
from orion.testing import create_experiment

config = dict(
    name="experiment-name",
    space={"x": "uniform(0, 200)"},
    metadata={
        "user": "test-user",
        "orion_version": "XYZ",
        "VCS": {
            "type": "git",
            "is_dirty": False,
            "HEAD_sha": "test",
            "active_branch": None,
            "diff_sha": "diff",
        },
    },
    version=1,
    pool_size=1,
    max_trials=10,
    working_dir="",
    algorithms={"random": {"seed": 1}},
    producer={"strategy": "NoParallelStrategy"},
)


trial_config = {
    "experiment": 0,
    "status": "new",  # new, reserved, suspended, completed, broken
    "worker": None,
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [],
    "params": [],
}


def check_regret_plot(plot):
    """Verifies that existence of the regret plot"""
    assert plot
    assert "regret" in plot.layout.title.text.lower()
    assert 2 == len(plot.data)


def test_init_require_experiment():
    """Tests that a `PlotAccessor` requires an instance of `ExperimentClient`"""
    with pytest.raises(ValueError) as exception:
        PlotAccessor(None)

    assert "Parameter 'experiment' is None" in str(exception.value)


def test_call_nonexistent_kind():
    """Tests that specifying a non existent kind will fail"""
    with create_experiment(config, trial_config, ["completed"]) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        with pytest.raises(ValueError) as exception:
            pa(kind="nonexistent")

        assert "Plot of kind 'nonexistent' is not one of" in str(exception.value)


def test_regret_is_default_plot():
    """Tests that the regret plot is the default plot"""
    with create_experiment(config, trial_config, ["completed"]) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        plot = pa()

        check_regret_plot(plot)


def test_regret_kind():
    """Tests that a regret plot can be created from specifying `kind` as a parameter."""
    with create_experiment(config, trial_config, ["completed"]) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        plot = pa(kind="regret")

        check_regret_plot(plot)


def test_call_to_regret():
    """Tests instance calls to `PlotAccessor.regret()`"""
    with create_experiment(config, trial_config, ["completed"]) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        plot = pa.regret()

        check_regret_plot(plot)
