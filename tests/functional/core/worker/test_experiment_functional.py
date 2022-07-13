#!/usr/bin/env python
"""Collection of functional tests for :mod:`orion.core.worker.experiment`."""
import logging

from orion.client import build_experiment, get_experiment
from orion.core.io.database import DuplicateKeyError
from orion.core.worker.trial import Trial
from orion.testing import mocked_datetime
from orion.testing.evc import (
    build_child_experiment,
    build_root_experiment,
    disable_duplication,
)

SPACE = {"x": "uniform(0, 100)"}
N_PENDING = 3  # new, interrupted and suspended


def generate_trials_list(level, statuses=Trial.allowed_stati):
    return [
        {"status": trial_status, "x": i + len(statuses) * level}
        for i, trial_status in enumerate(statuses)
    ]


status = []


def build_evc_tree(levels):
    build_root_experiment(space=SPACE, trials=generate_trials_list(levels[0]))
    names = ["root", "parent", "experiment", "child", "grand-child"]
    for level, (parent, name) in zip(levels[1:], zip(names, names[1:])):
        build_child_experiment(
            space=SPACE, name=name, parent=parent, trials=generate_trials_list(level)
        )


def test_duplicate_pending_trials(storage, monkeypatch):
    """Test that only pending trials are duplicated"""
    with disable_duplication(monkeypatch):
        build_evc_tree(list(range(5)))

    for exp in ["root", "parent", "experiment", "child", "grand-child"]:
        assert len(get_experiment(name=exp).fetch_trials(with_evc_tree=False)) == len(
            Trial.allowed_stati
        )

    experiment = build_experiment(name="experiment")
    experiment._experiment.duplicate_pending_trials()

    for exp in ["root", "parent", "child", "grand-child"]:
        assert len(get_experiment(name=exp).fetch_trials(with_evc_tree=False)) == len(
            Trial.allowed_stati
        )

    assert (
        len(experiment.fetch_trials(with_evc_tree=False))
        == len(Trial.allowed_stati) + N_PENDING * 4
    )


def test_duplicate_closest_duplicated_pending_trials(storage, monkeypatch):
    """Test that only closest duplicated pending trials are duplicated"""
    with disable_duplication(monkeypatch):
        build_evc_tree([0, 0, 1, 2, 2])

    for exp in ["root", "parent", "experiment", "child", "grand-child"]:
        assert len(get_experiment(name=exp).fetch_trials(with_evc_tree=False)) == len(
            Trial.allowed_stati
        )

    experiment = build_experiment(name="experiment")
    experiment._experiment.duplicate_pending_trials()

    for exp in ["root", "parent", "child", "grand-child"]:
        assert len(get_experiment(name=exp).fetch_trials(with_evc_tree=False)) == len(
            Trial.allowed_stati
        )

    assert (
        len(experiment.fetch_trials(with_evc_tree=False))
        == len(Trial.allowed_stati) + N_PENDING * 2
    )


def test_duplicate_only_once(storage, monkeypatch):
    """Test that trials may not be duplicated twice"""
    with disable_duplication(monkeypatch):
        build_evc_tree(list(range(5)))

    for exp in ["root", "parent", "experiment", "child", "grand-child"]:
        assert len(get_experiment(name=exp).fetch_trials(with_evc_tree=False)) == len(
            Trial.allowed_stati
        )

    experiment = build_experiment(name="experiment")
    experiment._experiment.duplicate_pending_trials()

    for exp in ["root", "parent", "child", "grand-child"]:
        assert len(get_experiment(name=exp).fetch_trials(with_evc_tree=False)) == len(
            Trial.allowed_stati
        )

    assert (
        len(experiment.fetch_trials(with_evc_tree=False))
        == len(Trial.allowed_stati) + N_PENDING * 4
    )

    experiment._experiment.duplicate_pending_trials()

    for exp in ["root", "parent", "child", "grand-child"]:
        assert len(get_experiment(name=exp).fetch_trials(with_evc_tree=False)) == len(
            Trial.allowed_stati
        )

    assert (
        len(experiment.fetch_trials(with_evc_tree=False))
        == len(Trial.allowed_stati) + N_PENDING * 4
    )


def test_duplicate_race_conditions(storage, monkeypatch, caplog):
    """Test that duplication does not raise an error during race conditions."""
    with disable_duplication(monkeypatch):
        build_evc_tree(list(range(2)))

    experiment = build_experiment(name="parent")

    def register_race_condition(trial):
        raise DuplicateKeyError("Race condition!")

    monkeypatch.setattr(
        experiment._experiment._storage, "register_trial", register_race_condition
    )

    assert len(experiment.fetch_trials(with_evc_tree=False)) == len(Trial.allowed_stati)

    with caplog.at_level(logging.DEBUG):
        experiment._experiment.duplicate_pending_trials()

        assert "Race condition while trying to duplicate trial" in caplog.text


def test_fix_lost_trials_in_evc(storage, monkeypatch):
    """Test that lost trials from parents can be fixed as well.

    `fix_lost_trials` is tested more carefully in experiment's unit-tests (without the EVC).
    """
    with disable_duplication(monkeypatch), mocked_datetime(monkeypatch):
        build_evc_tree(list(range(5)))

    for exp_name in ["root", "parent", "experiment", "child", "grand-child"]:
        exp = get_experiment(name=exp_name)
        assert len(exp.fetch_trials(with_evc_tree=False)) == len(Trial.allowed_stati)
        assert len(exp.fetch_trials_by_status("reserved", with_evc_tree=False)) == 1

    experiment = build_experiment(name="experiment")
    experiment._experiment.fix_lost_trials()

    for exp_name in ["root", "parent", "experiment", "child", "grand-child"]:
        exp = get_experiment(name=exp_name)
        assert len(exp.fetch_trials(with_evc_tree=False)) == len(Trial.allowed_stati)
        assert len(exp.fetch_trials_by_status("reserved", with_evc_tree=False)) == 0
