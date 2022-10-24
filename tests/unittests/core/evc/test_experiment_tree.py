#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.evc.experiment`."""

import pytest

from orion.client import get_experiment
from orion.testing.evc import (
    build_child_experiment,
    build_grand_child_experiment,
    build_root_experiment,
    disable_duplication,
)

ROOT_SPACE_WITH_DEFAULTS = {
    "x": "uniform(0, 100, default_value=0)",
    "y": "uniform(0, 100,  default_value=2)",
    "z": "uniform(0, 100, default_value=4)",
}

CHILD_SPACE_WITH_DEFAULTS = {
    "x": "uniform(0, 100, default_value=0)",
    "y": "uniform(0, 100,  default_value=2)",
}

GRAND_CHILD_SPACE_WITH_DEFAULTS = {
    "x": "uniform(0, 100, default_value=0)",
}


CHILD_SPACE_DELETION = {
    "x": "uniform(0, 100)",
    "y": "uniform(0, 100)",
}

GRAND_CHILD_SPACE_DELETION = {
    "x": "uniform(0, 100)",
}


CHILD_SPACE_PRIOR_CHANGE = {
    "x": "uniform(0, 8)",
    "y": "uniform(0, 8)",
    "z": "uniform(0, 8)",
}

GRAND_CHILD_SPACE_PRIOR_CHANGE = {
    "x": "uniform(0, 3)",
    "y": "uniform(0, 3)",
    "z": "uniform(0, 3)",
}


CHILD_TRIALS_DUPLICATES = [{"x": i, "y": i * 2, "z": i**2} for i in range(2, 8)]

GRAND_CHILD_TRIALS_DUPLICATES = [
    {"x": i, "y": i * 2, "z": i**2} for i in list(range(1, 4)) + list(range(8, 10))
]


CHILD_TRIALS_DELETION = [{"x": i, "y": i * 2} for i in range(4, 10)]

GRAND_CHILD_TRIALS_DELETION = [{"x": i} for i in range(10, 15)]


CHILD_TRIALS_PRIOR_CHANGE = [{"x": i, "y": i / 2, "z": i / 4} for i in range(1, 8)]

GRAND_CHILD_TRIALS_PRIOR_CHANGE = [
    {"x": i * 2 / 10, "y": i / 10, "z": i / 20} for i in range(1, 10)
]


def generic_tree_test(
    experiment_name,
    parent_name=None,
    grand_parent_name=None,
    children_names=tuple(),
    grand_children_names=tuple(),
    node_trials=0,
    parent_trials=0,
    grand_parent_trials=0,
    children_trials=tuple(),
    grand_children_trials=tuple(),
    total_trials=0,
):
    """Test fetching of trials from experiments in the EVC tree.

    Parameters
    ----------
    experiment_name: str
        The name of the experiment that will be the main node for the tests.
    parent_name: str or None
        The name of the parent experiment, this will be used to fetch the trials from the parent
        experiment directly (not in EVC) for comparison.
    grand_parent_name: str or None
        The name of the grand parent experiment, this will be used to fetch the trials from the
        grand parent experiment directly (not in EVC) for comparison.
    children_names: list or str
        The names of the children experiments, this will be used to fetch the trials from the
        children experiments directly (not in EVC) for comparison.
    grand_children_names: list or str
        The names of the grand children experiments, this will be used to fetch the trials from the
        grand children experiments directly (not in EVC) for comparison. All grand children names
        may be included in the list even though they are associated to different children.
    node_trials: int,
        The number of trials that should be fetched from current node experiment.
    parent_trials: int,
        The number of trials that should be fetched from parent experiment (not using EVC tree).
    grand_parent_trials: int,
        The number of trials that should be fetched from grand parent experiment (not using EVC tree).
    children_trials: list of int,
        The number of trials that should be fetched from each children experiment (not using EVC tree).
    grand_children_trials: list of int,
        The number of trials that should be fetched from each grand children experiment (not using EVC tree).
    total_trials: int,
        The number of trials that should be fetched from current node experiment when fetching
        recursively from the EVC tree. This may not be equal to the sum of all trials in parent and
        children experiments depending on the adapters.

    """

    experiment = get_experiment(experiment_name)
    exp_node = experiment.node

    assert exp_node.item.name == experiment_name

    num_nodes = 1

    if parent_name:
        assert exp_node.parent.item.name == parent_name
        num_nodes += 1
    if grand_parent_name:
        assert exp_node.parent.parent.item.name == grand_parent_name
        num_nodes += 1

    assert len(exp_node.children) == len(children_names)
    if children_names:
        assert [child.item.name for child in exp_node.children] == children_names
        num_nodes += len(children_names)

    if grand_children_names:
        grand_children = sum((child.children for child in exp_node.children), [])
        assert [child.item.name for child in grand_children] == grand_children_names
        num_nodes += len(grand_children_names)

    assert len(list(exp_node.root)) == num_nodes

    print("In node")
    for trial in experiment.fetch_trials():
        print(trial)
    assert len(experiment.fetch_trials()) == node_trials
    if parent_name:
        print("In parent")
        for trial in exp_node.parent.item.fetch_trials():
            print(trial)
        assert len(exp_node.parent.item.fetch_trials()) == parent_trials
    if grand_parent_name:
        print("In grand-parent")
        for trial in exp_node.parent.parent.item.fetch_trials():
            print(trial)
        assert len(exp_node.parent.parent.item.fetch_trials()) == grand_parent_trials

    if children_names:
        print("In children")
        for trial in exp_node.children[0].item.fetch_trials():
            print(trial)
        assert [
            len(child.item.fetch_trials()) for child in exp_node.children
        ] == children_trials

    if grand_children_names:
        grand_children = sum((child.children for child in exp_node.children), [])
        all_trials = sum(
            (child_node.item.fetch_trials() for child_node in grand_children), []
        )
        print("In grand-children")
        for trial in all_trials:
            print(trial)
        assert [
            len(child.item.fetch_trials()) for child in grand_children
        ] == grand_children_trials

    print("with evc")
    for trial in experiment.fetch_trials(with_evc_tree=True):
        print(trial)

    assert len(experiment.fetch_trials(with_evc_tree=True)) == total_trials

    all_ids = [trial.id for trial in experiment.fetch_trials(with_evc_tree=True)]
    exp_ids = [trial.id for trial in experiment.fetch_trials(with_evc_tree=False)]

    # Ensure all trials of experiment are fetched when fetching from all EVC
    # It could happen that some trials are missing if duplicates are incorrectly filtered out
    # from current node instead of from parent or child.
    assert set(exp_ids) - set(all_ids) == set()


parametrization = {
    "no-adapter-parent": (
        {},
        {},
        None,
        dict(
            experiment_name="child",
            parent_name="root",
            node_trials=6,
            parent_trials=4,
            total_trials=10,
        ),
    ),
    "no-adapter-children": (
        {},
        {},
        None,
        dict(
            experiment_name="root",
            children_names=["child"],
            node_trials=4,
            children_trials=[6],
            total_trials=10,
        ),
    ),
    "no-adapter-parent-children": (
        {},
        {},
        {},
        dict(
            experiment_name="child",
            parent_name="root",
            children_names=["grand-child"],
            node_trials=6,
            parent_trials=4,
            children_trials=[5],
            total_trials=15,
        ),
    ),
    "no-adapter-parent-parent": (
        {},
        {},
        {},
        dict(
            experiment_name="grand-child",
            parent_name="child",
            grand_parent_name="root",
            node_trials=5,
            parent_trials=6,
            grand_parent_trials=4,
            total_trials=15,
        ),
    ),
    "no-adapter-children-children": (
        {},
        {},
        {},
        dict(
            experiment_name="root",
            children_names=["child"],
            grand_children_names=["grand-child"],
            node_trials=4,
            children_trials=[6],
            grand_children_trials=[5],
            total_trials=15,
        ),
    ),
    "duplicates-parent": (
        {},
        dict(trials=CHILD_TRIALS_DUPLICATES),
        None,
        dict(
            experiment_name="child",
            parent_name="root",
            node_trials=6,
            parent_trials=4,
            total_trials=8,
        ),
    ),
    "duplicates-children": (
        {},
        dict(trials=CHILD_TRIALS_DUPLICATES),
        None,
        dict(
            experiment_name="root",
            children_names=["child"],
            node_trials=4,
            children_trials=[6],
            total_trials=8,
        ),
    ),
    "duplicates-parent-children": (
        {},
        dict(trials=CHILD_TRIALS_DUPLICATES),
        dict(trials=GRAND_CHILD_TRIALS_DUPLICATES),
        dict(
            experiment_name="child",
            parent_name="root",
            children_names=["grand-child"],
            node_trials=6,
            parent_trials=4,
            children_trials=[5],
            total_trials=6
            + 1  # Only 1 trial from root
            + 1  # 1 trial from grand_child with i=1
            + 2,  # 2 trials from grand_child with i>=8,
        ),
    ),
    "duplicates-parent-parent": (
        {},
        dict(trials=CHILD_TRIALS_DUPLICATES),
        dict(trials=GRAND_CHILD_TRIALS_DUPLICATES),
        dict(
            experiment_name="grand-child",
            parent_name="child",
            grand_parent_name="root",
            node_trials=5,
            parent_trials=6,
            grand_parent_trials=4,
            total_trials=5
            + 4  # 4 trials from `child` experiment (parent)
            + 1,  # 1 trial from `root` experiment (grand-parent)
        ),
    ),
    "duplicates-children-children": (
        {},
        dict(trials=CHILD_TRIALS_DUPLICATES),
        dict(trials=GRAND_CHILD_TRIALS_DUPLICATES),
        dict(
            experiment_name="root",
            children_names=["child"],
            grand_children_names=["grand-child"],
            node_trials=4,
            children_trials=[6],
            grand_children_trials=[5],
            total_trials=4
            + 4  # 4 trials from `child` experiment
            + 2,  # 2 trials from  `grand-child` experiment
        ),
    ),
    "deletion-with-default-forward": (
        dict(space=ROOT_SPACE_WITH_DEFAULTS),
        dict(space=CHILD_SPACE_WITH_DEFAULTS),
        None,
        dict(
            experiment_name="child",
            parent_name="root",
            node_trials=6,
            parent_trials=4,
            total_trials=7,
        ),
    ),
    "deletion-with-default-backward": (
        dict(space=ROOT_SPACE_WITH_DEFAULTS),
        dict(space=CHILD_SPACE_WITH_DEFAULTS),
        None,
        dict(
            experiment_name="root",
            children_names=["child"],
            node_trials=4,
            children_trials=[6],
            total_trials=10,
        ),
    ),
    "deletion-without-default-forward": (
        dict(),
        dict(space=CHILD_SPACE_DELETION),
        None,
        dict(
            experiment_name="child",
            parent_name="root",
            node_trials=6,
            parent_trials=4,
            total_trials=6,
        ),
    ),
    "deletion-without-default-backward": (
        dict(),
        dict(space=CHILD_SPACE_DELETION),
        None,
        dict(
            experiment_name="root",
            children_names=["child"],
            node_trials=4,
            children_trials=[6],
            total_trials=4,
        ),
    ),
    "deletion-with-default-forward-forward": (
        dict(space=ROOT_SPACE_WITH_DEFAULTS),
        dict(space=CHILD_SPACE_WITH_DEFAULTS, trials=CHILD_TRIALS_DELETION),
        dict(space=GRAND_CHILD_SPACE_WITH_DEFAULTS),
        dict(
            experiment_name="grand-child",
            parent_name="child",
            grand_parent_name="root",
            node_trials=5,
            parent_trials=6,
            grand_parent_trials=4,
            total_trials=5,
        ),
    ),
    "deletion-with-default-forward-backward": (
        dict(space=ROOT_SPACE_WITH_DEFAULTS),
        dict(space=CHILD_SPACE_WITH_DEFAULTS, trials=CHILD_TRIALS_DELETION),
        dict(space=GRAND_CHILD_SPACE_WITH_DEFAULTS),
        dict(
            experiment_name="child",
            parent_name="root",
            children_names=["grand-child"],
            node_trials=6,
            parent_trials=4,
            children_trials=[5],
            total_trials=6 + 1 + 5,
        ),
    ),
    "deletion-with-default-backward-backward": (
        dict(space=ROOT_SPACE_WITH_DEFAULTS),
        dict(space=CHILD_SPACE_WITH_DEFAULTS, trials=CHILD_TRIALS_DELETION),
        dict(space=GRAND_CHILD_SPACE_WITH_DEFAULTS),
        dict(
            experiment_name="root",
            children_names=["child"],
            grand_children_names=["grand-child"],
            node_trials=4,
            children_trials=[6],
            grand_children_trials=[5],
            total_trials=4 + 6 + 5,
        ),
    ),
    "deletion-without-default-forward-forward": (
        dict(),
        dict(space=CHILD_SPACE_DELETION, trials=CHILD_TRIALS_DELETION),
        dict(space=GRAND_CHILD_SPACE_DELETION),
        dict(
            experiment_name="grand-child",
            parent_name="child",
            grand_parent_name="root",
            node_trials=5,
            parent_trials=6,
            grand_parent_trials=4,
            total_trials=5,
        ),
    ),
    "deletion-without-default-forward-backward": (
        dict(),
        dict(space=CHILD_SPACE_DELETION, trials=CHILD_TRIALS_DELETION),
        dict(space=GRAND_CHILD_SPACE_DELETION),
        dict(
            experiment_name="child",
            parent_name="root",
            children_names=["grand-child"],
            node_trials=6,
            parent_trials=4,
            children_trials=[5],
            total_trials=6,
        ),
    ),
    "deletion-without-default-backward-backward": (
        dict(),
        dict(space=CHILD_SPACE_DELETION, trials=CHILD_TRIALS_DELETION),
        dict(space=GRAND_CHILD_SPACE_DELETION),
        dict(
            experiment_name="root",
            children_names=["child"],
            grand_children_names=["grand-child"],
            node_trials=4,
            children_trials=[6],
            grand_children_trials=[5],
            total_trials=4,
        ),
    ),
    "prior-change-forward": (
        dict(),
        dict(space=CHILD_SPACE_PRIOR_CHANGE, trials=CHILD_TRIALS_PRIOR_CHANGE),
        None,
        dict(
            experiment_name="child",
            parent_name="root",
            node_trials=len(CHILD_TRIALS_PRIOR_CHANGE),
            parent_trials=4,
            total_trials=len(CHILD_TRIALS_PRIOR_CHANGE) + 4 - 1,  # One is out of bound
        ),
    ),
    "prior-change-backward": (
        dict(),
        dict(space=CHILD_SPACE_PRIOR_CHANGE, trials=CHILD_TRIALS_PRIOR_CHANGE),
        None,
        dict(
            experiment_name="root",
            children_names=["child"],
            node_trials=4,
            children_trials=[len(CHILD_TRIALS_PRIOR_CHANGE)],
            total_trials=len(CHILD_TRIALS_PRIOR_CHANGE) + 4,  # They are all included
        ),
    ),
    "prior-change-forward-forward": (
        dict(),
        dict(space=CHILD_SPACE_PRIOR_CHANGE, trials=CHILD_TRIALS_PRIOR_CHANGE),
        dict(
            space=GRAND_CHILD_SPACE_PRIOR_CHANGE, trials=GRAND_CHILD_TRIALS_PRIOR_CHANGE
        ),
        dict(
            experiment_name="grand-child",
            parent_name="child",
            grand_parent_name="root",
            node_trials=len(GRAND_CHILD_TRIALS_PRIOR_CHANGE),
            parent_trials=len(CHILD_TRIALS_PRIOR_CHANGE),
            grand_parent_trials=4,
            total_trials=len(GRAND_CHILD_TRIALS_PRIOR_CHANGE)
            + sum(trial["x"] <= 3 for trial in CHILD_TRIALS_PRIOR_CHANGE)
            + 2,  # Only 2 of root trials are compatible with grand-child space
        ),
    ),
    "prior-change-backward-forward": (
        dict(),
        dict(space=CHILD_SPACE_PRIOR_CHANGE, trials=CHILD_TRIALS_PRIOR_CHANGE),
        dict(
            space=GRAND_CHILD_SPACE_PRIOR_CHANGE, trials=GRAND_CHILD_TRIALS_PRIOR_CHANGE
        ),
        dict(
            experiment_name="child",
            parent_name="root",
            children_names=["grand-child"],
            node_trials=len(CHILD_TRIALS_PRIOR_CHANGE),
            parent_trials=4,
            children_trials=[len(GRAND_CHILD_TRIALS_PRIOR_CHANGE)],
            total_trials=len(GRAND_CHILD_TRIALS_PRIOR_CHANGE)
            + len(CHILD_TRIALS_PRIOR_CHANGE)  # All trials are compatible
            + 3,  # Only 3 of root trials are compatible with grand-child space
        ),
    ),
    "prior-change-backward-backward": (
        dict(),
        dict(space=CHILD_SPACE_PRIOR_CHANGE, trials=CHILD_TRIALS_PRIOR_CHANGE),
        dict(
            space=GRAND_CHILD_SPACE_PRIOR_CHANGE, trials=GRAND_CHILD_TRIALS_PRIOR_CHANGE
        ),
        dict(
            experiment_name="root",
            children_names=["child"],
            grand_children_names=["grand-child"],
            node_trials=4,
            children_trials=[len(CHILD_TRIALS_PRIOR_CHANGE)],
            grand_children_trials=[len(GRAND_CHILD_TRIALS_PRIOR_CHANGE)],
            total_trials=len(GRAND_CHILD_TRIALS_PRIOR_CHANGE)
            + len(CHILD_TRIALS_PRIOR_CHANGE)
            + 4,  # All trials are compatible
        ),
    ),
}


@pytest.mark.parametrize(
    "root, child, grand_child, test_kwargs",
    list(parametrization.values()),
    ids=list(parametrization.keys()),
)
def test_evc_fetch_adapters(
    monkeypatch, storage, root, child, grand_child, test_kwargs
):
    """Test the recursive fetch of trials in the EVC tree."""
    with disable_duplication(monkeypatch):
        build_root_experiment(**root)
        build_child_experiment(**child)
        if grand_child is not None:
            build_grand_child_experiment(**grand_child)
    generic_tree_test(**test_kwargs)
