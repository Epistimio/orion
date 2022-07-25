""" Tests for the experiment config dataclasses. """
from orion.client import build_experiment
from orion.core.worker.experiment_config import ExperimentConfig


def test_can_be_created_from_exp():
    """An ExperimentConfig can be created from the `configuration` property of an Experiment."""
    experiment = build_experiment(name="foo", space={"x": "uniform(0, 5)"}, debug=True)
    exp: ExperimentConfig = experiment.configuration
    # TODO: Check that all fields match their type annotations.
    assert exp == experiment.configuration
