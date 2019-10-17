#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.utils.format`."""

import pytest

from orion.core.utils.format_trials import (trial_to_tuple, tuple_to_trial)
from orion.core.worker.trial import Trial


@pytest.fixture()
def trial():
    """Stab trial to match tuple from fixture `fixed_suggestion`."""
    params = dict(
        yolo=dict(
            name='yolo',
            type='categorical',
            value=('asdfa', 2)
            ),
        yolo2=dict(
            name='yolo2',
            type='integer',
            value=0
            ),
        yolo3=dict(
            name='yolo3',
            type='real',
            value=3.5
            ))
    return Trial(params=params)


def test_trial_to_tuple(space, trial, fixed_suggestion):
    """Check if trial is correctly created from a sample/tuple."""
    data = trial_to_tuple(trial, space)
    assert data == fixed_suggestion

    print(trial.params)
    trial.params['yolo'].name = 'lalala'
    with pytest.raises(ValueError) as exc:
        trial_to_tuple(trial, space)

    assert "Trial params: [\'lalala\', \'yolo2\', \'yolo3\']" in str(exc.value)

    trial.params.pop('yolo')
    with pytest.raises(ValueError) as exc:
        trial_to_tuple(trial, space)

    assert "Trial params: [\'yolo2\', \'yolo3\']" in str(exc.value)


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
    for key in t.params:
        assert t.params[key].to_dict() == trial.params[key].to_dict()


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
    for key in t.params:
        assert t.params[key].to_dict() == trial.params[key].to_dict()
