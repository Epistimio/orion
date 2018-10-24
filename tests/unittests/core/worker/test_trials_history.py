#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.trials_history`."""

from orion.core.worker.trials_history import TrialsHistory


class DummyTrial(object):
    """Mocking class for the Trial"""

    def __init__(self, trial_id, parents):
        """Init _id and parents only"""
        self._id = trial_id
        self.parents = parents


def test_added_children_without_ancestors():
    """Verify that children are added to history"""
    trials_history = TrialsHistory()
    trials_history.update([DummyTrial(i, []) for i in range(3)])
    assert trials_history.children == [0, 1, 2]
    trials_history.update([DummyTrial(i, []) for i in range(3, 6)])
    assert trials_history.children == [0, 1, 2, 3, 4, 5]


def test_added_children_with_ancestors():
    """Verify that children with ancestors are added to history"""
    trials_history = TrialsHistory()
    trials = [DummyTrial(i, []) for i in range(3)]
    trials_history.update(trials)
    assert trials_history.children == [0, 1, 2]

    trials = [DummyTrial(i, [trials[i % 3]._id]) for i in range(3, 6)]
    trials_history.update(trials)
    assert len(set(trials_history.children) & set([3, 4, 5])) == 3


def test_discarded_children():
    """Verify that ancestors of new children are discarded from history"""
    trials_history = TrialsHistory()
    trials = [DummyTrial(i, []) for i in range(3)]
    trials_history.update(trials)
    assert trials_history.children == [0, 1, 2]

    trials_history.update(trials)
    assert trials_history.children == [0, 1, 2]

    trials = [DummyTrial(i, [trials[i % 3]._id]) for i in range(3, 6)]
    trials_history.update(trials)
    assert trials_history.children == [3, 4, 5]


def test_discarded_duplicate_children():
    """Verify that duplicate children are not added twice"""
    trials_history = TrialsHistory()
    trials = [DummyTrial(i, []) for i in range(3)]
    trials_history.update(trials)
    assert trials_history.children == [0, 1, 2]

    trials = [DummyTrial(i, [trials[i]._id]) for i in range(3)]
    assert all(trial._id == trial.parents[0] for trial in trials)
    trials_history.update(trials)
    assert trials_history.children == [0, 1, 2]


def test_discarded_shared_children():
    """Verify that only ancestors are removed and not all past children"""
    trials_history = TrialsHistory()
    trials = [DummyTrial(i, []) for i in range(3)]
    trials_history.update(trials)
    assert trials_history.children == [0, 1, 2]

    trials = [DummyTrial(i, [0]) for i in range(3, 6)]
    trials_history.update(trials)
    assert trials_history.children == [1, 2, 3, 4, 5]
