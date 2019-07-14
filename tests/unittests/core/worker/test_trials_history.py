#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.trials_history`."""

from orion.core.worker.trials_history import TrialsHistory


class DummyTrial(object):
    """Mocking class for the Trial"""

    def __init__(self, trial_id, parents):
        """Init _id and parents only"""
        self.id = trial_id
        self.parents = parents


def test_history_contains_new_child():
    """Verify that __contains__ return True for a new child"""
    trials_history = TrialsHistory()
    new_child = DummyTrial(1, [])
    assert new_child not in trials_history
    trials_history.update([new_child])
    assert new_child in trials_history


def test_history_contains_old_child():
    """Verify that __contains__ return True for a new child"""
    trials_history = TrialsHistory()
    old_child = DummyTrial(1, [])
    trials_history.update([old_child])
    new_child = DummyTrial(2, [old_child.id])
    assert new_child not in trials_history
    trials_history.update([new_child])
    assert old_child.id not in trials_history.children
    assert old_child in trials_history
    assert new_child.id in trials_history.children
    assert new_child in trials_history


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

    trials = [DummyTrial(i, [trials[i % 3].id]) for i in range(3, 6)]
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

    trials = [DummyTrial(i, [trials[i % 3].id]) for i in range(3, 6)]
    trials_history.update(trials)
    assert trials_history.children == [3, 4, 5]


def test_discarded_duplicate_children():
    """Verify that duplicate children are not added twice"""
    trials_history = TrialsHistory()
    trials = [DummyTrial(i, []) for i in range(3)]
    trials_history.update(trials)
    assert trials_history.children == [0, 1, 2]

    trials = [DummyTrial(i, [trials[i].id]) for i in range(3)]
    assert all(trial.id == trial.parents[0] for trial in trials)
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
