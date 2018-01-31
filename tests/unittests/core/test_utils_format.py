#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`metaopt.core.utils.format`."""

import pytest

from metaopt.core.worker.trial import Trial
from metaopt.core.utils.format import (tuple_to_trial, trial_to_tuple)


@pytest.fixture()
def trial():
    """Stab trial to match tuple from fixture `fixed_suggestion`."""
    params = [
        dict(
            name='yolo',
            type='categorical',
            value=('asdfa', 2)
            ),
        dict(
            name='yolo2',
            type='integer',
            value=0
            ),
        dict(
            name='yolo3',
            type='real',
            value=3.5
            )
        ]
    return Trial(params=params)


def test_trial_to_tuple(space, trial, fixed_suggestion):
    """Check if trial is correctly created from a sample/tuple."""
    data = trial_to_tuple(trial, space)
    assert data == fixed_suggestion

    trial.params[0].name = 'lalala'
    with pytest.raises(AssertionError):
        trial_to_tuple(trial, space)
    trial.params.pop(0)
    with pytest.raises(AssertionError):
        trial_to_tuple(trial, space)


def test_tuple_to_trial(space, trial, fixed_suggestion):
    """Check if sample is recovered successfully from trial."""
    t = tuple_to_trial(fixed_suggestion, space)
    assert t.experiment is None
    assert t.status == 'new'
    assert t.worker is None
    assert t.submit_time is None
    assert t.start_time is None
    assert t.end_time is None
    assert t.results == []
    assert len(t.params) == len(trial.params)
    for i in range(len(t.params)):
        assert t.params[i].to_dict() == trial.params[i].to_dict()


def test_tuple_to_trial_to_tuple(space, trial, fixed_suggestion):
    """The two functions should be inverse."""
    data = trial_to_tuple(tuple_to_trial(fixed_suggestion, space), space)
    assert data == fixed_suggestion

    t = tuple_to_trial(trial_to_tuple(trial, space), space)
    assert t.experiment is None
    assert t.status == 'new'
    assert t.worker is None
    assert t.submit_time is None
    assert t.start_time is None
    assert t.end_time is None
    assert t.results == []
    assert len(t.params) == len(trial.params)
    for i in range(len(t.params)):
        assert t.params[i].to_dict() == trial.params[i].to_dict()
