""" Tests for the experiment config dataclasses. """

from orion.client import build_experiment
from orion.core.worker.experiment_config import ExperimentConfig


def test_fields_match_exp_config_dict():
    """Test that the annotations on `ExperimentConfig`'s match the Experiment config dict."""
    experiment = build_experiment(name="foo", space={"x": "uniform(0, 5)"}, debug=True)
    exp: ExperimentConfig = experiment.configuration
    assert exp == experiment.configuration

    actual_config_dict = experiment.configuration
    # NOTE: We don't actually check the type of the value, because we would need to evaluate
    # the forward references, and those might contain things like `int | None` that we can't
    # evaluate.
    assert set(actual_config_dict.keys()) == set(
        ExperimentConfig.__annotations__.keys()
    )
