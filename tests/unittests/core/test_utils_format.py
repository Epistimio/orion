#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.core.utils.format`."""

import pytest

from orion.core.utils.format_trials import dict_to_trial, trial_to_tuple, tuple_to_trial
from orion.core.worker.trial import Trial


@pytest.fixture()
def params_tuple():
    """Stab param tuple to match trial from fixture `fixed_suggestion`."""
    return (("asdfa", 2), 0, 3.5)


@pytest.fixture()
def hierarchical_trial():
    """Stab trial with hierarchical params."""
    params = [
        dict(name="yolo.first", type="categorical", value=("asdfa", 2)),
        dict(name="yolo.second", type="integer", value=0),
        dict(name="yoloflat", type="real", value=3.5),
    ]
    return Trial(params=params)


@pytest.fixture()
def dict_params():
    """Return dictionary of params to build a trial like `fixed_suggestion`"""
    return {"yolo": ("asdfa", 2), "yolo2": 0, "yolo3": 3.5}


@pytest.fixture()
def hierarchical_dict_params():
    """Return dictionary of params to build a hierarchical trial"""
    return {"yolo": {"first": ("asdfa", 2), "second": 0}, "yoloflat": 3.5}


def test_trial_to_tuple(space, fixed_suggestion, params_tuple):
    """Check if trial is correctly created from a sample/tuple."""
    data = trial_to_tuple(fixed_suggestion, space)
    assert data == params_tuple

    fixed_suggestion._params[0].name = "lalala"
    with pytest.raises(ValueError) as exc:
        trial_to_tuple(fixed_suggestion, space)

    assert "Trial params: ['lalala', 'yolo2', 'yolo3']" in str(exc.value)

    fixed_suggestion._params.pop(0)
    with pytest.raises(ValueError) as exc:
        trial_to_tuple(fixed_suggestion, space)

    assert "Trial params: ['yolo2', 'yolo3']" in str(exc.value)


def test_tuple_to_trial(space, fixed_suggestion, params_tuple):
    """Check if sample is recovered successfully from trial."""
    t = tuple_to_trial(params_tuple, space)
    assert t.experiment is None
    assert t.status == "new"
    assert t.worker is None
    assert t.submit_time is None
    assert t.start_time is None
    assert t.end_time is None
    assert t.results == []
    assert len(t._params) == len(fixed_suggestion.params)
    for i in range(len(t.params)):
        assert t._params[i].to_dict() == fixed_suggestion._params[i].to_dict()


def test_dict_to_trial(space, fixed_suggestion, dict_params):
    """Check if dict is converted successfully to trial."""
    t = dict_to_trial(dict_params, space)
    assert t.experiment is None
    assert t.status == "new"
    assert t.worker is None
    assert t.submit_time is None
    assert t.start_time is None
    assert t.end_time is None
    assert t.results == []
    assert len(t._params) == len(fixed_suggestion._params)
    for i in range(len(t.params)):
        assert t._params[i].to_dict() == fixed_suggestion._params[i].to_dict()


def test_tuple_to_trial_to_tuple(space, fixed_suggestion, params_tuple):
    """The two functions should be inverse."""
    data = trial_to_tuple(tuple_to_trial(params_tuple, space), space)
    assert data == params_tuple

    t = tuple_to_trial(trial_to_tuple(fixed_suggestion, space), space)
    assert t.experiment is None
    assert t.status == "new"
    assert t.worker is None
    assert t.submit_time is None
    assert t.start_time is None
    assert t.end_time is None
    assert t.results == []
    assert len(t._params) == len(fixed_suggestion._params)
    for i in range(len(t._params)):
        assert t._params[i].to_dict() == fixed_suggestion._params[i].to_dict()


def test_hierarchical_trial_to_tuple(
    hierarchical_space, hierarchical_trial, params_tuple
):
    """Check if hierarchical trial is correctly created from a sample/tuple."""
    data = trial_to_tuple(hierarchical_trial, hierarchical_space)
    assert data == params_tuple


def test_tuple_to_hierarchical_trial(
    hierarchical_space, hierarchical_trial, params_tuple
):
    """Check if sample is recovered successfully from hierarchical trial."""
    t = tuple_to_trial(params_tuple, hierarchical_space)
    assert len(t._params) == len(hierarchical_trial._params)
    for i in range(len(t._params)):
        assert t._params[i].to_dict() == hierarchical_trial._params[i].to_dict()


def test_hierarchical_dict_to_trial(
    hierarchical_space, hierarchical_trial, hierarchical_dict_params
):
    """Check if hierarchical dict is converted successfully to trial."""
    t = dict_to_trial(hierarchical_dict_params, hierarchical_space)
    assert len(t._params) == len(hierarchical_trial._params)
    for i in range(len(t.params)):
        assert t._params[i].to_dict() == hierarchical_trial._params[i].to_dict()
