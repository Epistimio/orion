#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.utils.format`."""

import pytest

from orion.core.utils.format_trials import dict_to_trial, trial_to_tuple, tuple_to_trial
from orion.core.worker.trial import Trial


@pytest.fixture()
def trial():
    """Stab trial to match tuple from fixture `fixed_suggestion`."""
    params = [
        dict(name="yolo", type="categorical", value=("asdfa", 2)),
        dict(name="yolo2", type="integer", value=0),
        dict(name="yolo3", type="real", value=3.5),
    ]
    return Trial(params=params)


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


def test_trial_to_tuple(space, trial, fixed_suggestion):
    """Check if trial is correctly created from a sample/tuple."""
    data = trial_to_tuple(trial, space)
    assert data == fixed_suggestion

    trial._params[0].name = "lalala"
    with pytest.raises(ValueError) as exc:
        trial_to_tuple(trial, space)

    assert "Trial params: ['lalala', 'yolo2', 'yolo3']" in str(exc.value)

    trial._params.pop(0)
    with pytest.raises(ValueError) as exc:
        trial_to_tuple(trial, space)

    assert "Trial params: ['yolo2', 'yolo3']" in str(exc.value)


def test_tuple_to_trial(space, trial, fixed_suggestion):
    """Check if sample is recovered successfully from trial."""
    t = tuple_to_trial(fixed_suggestion, space)
    assert t.experiment is None
    assert t.status == "new"
    assert t.worker is None
    assert t.submit_time is None
    assert t.start_time is None
    assert t.end_time is None
    assert t.results == []
    assert len(t._params) == len(trial.params)
    for i in range(len(t.params)):
        assert t._params[i].to_dict() == trial._params[i].to_dict()


def test_dict_to_trial(space, trial, dict_params):
    """Check if dict is converted successfully to trial."""
    t = dict_to_trial(dict_params, space)
    assert t.experiment is None
    assert t.status == "new"
    assert t.worker is None
    assert t.submit_time is None
    assert t.start_time is None
    assert t.end_time is None
    assert t.results == []
    assert len(t._params) == len(trial._params)
    for i in range(len(t.params)):
        assert t._params[i].to_dict() == trial._params[i].to_dict()


def test_tuple_to_trial_to_tuple(space, trial, fixed_suggestion):
    """The two functions should be inverse."""
    data = trial_to_tuple(tuple_to_trial(fixed_suggestion, space), space)
    assert data == fixed_suggestion

    t = tuple_to_trial(trial_to_tuple(trial, space), space)
    assert t.experiment is None
    assert t.status == "new"
    assert t.worker is None
    assert t.submit_time is None
    assert t.start_time is None
    assert t.end_time is None
    assert t.results == []
    assert len(t._params) == len(trial._params)
    for i in range(len(t._params)):
        assert t._params[i].to_dict() == trial._params[i].to_dict()


def test_hierarchical_trial_to_tuple(
    hierarchical_space, hierarchical_trial, fixed_suggestion
):
    """Check if hierarchical trial is correctly created from a sample/tuple."""
    data = trial_to_tuple(hierarchical_trial, hierarchical_space)
    assert data == fixed_suggestion


def test_tuple_to_hierarchical_trial(
    hierarchical_space, hierarchical_trial, fixed_suggestion
):
    """Check if sample is recovered successfully from hierarchical trial."""
    t = tuple_to_trial(fixed_suggestion, hierarchical_space)
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
