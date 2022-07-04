""" Tests for the experiment config dataclasses. """
from orion.client import build_experiment
from orion.core.worker.warm_start.experiment_config import ExperimentInfo


def test_can_be_created_from_exp():
    """An ExperimentInfo can be created from the `configuration` property of an Experiment."""
    experiment = build_experiment(name="foo", space={"x": "uniform(0, 5)"}, debug=True)
    exp = ExperimentInfo.from_dict(experiment.configuration)
    assert exp.to_dict() == experiment.configuration
