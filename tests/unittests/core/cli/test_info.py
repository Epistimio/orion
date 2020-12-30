#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.cli.info`."""
import itertools

import pytest

from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.format_terminal import (
    format_algorithm,
    format_commandline,
    format_config,
    format_dict,
    format_identification,
    format_info,
    format_list,
    format_metadata,
    format_refers,
    format_space,
    format_stats,
    format_title,
    get_trial_params,
)
from orion.core.worker.trial import Trial


class DummyExperiment:
    """Dummy container to mock experiments"""

    pass


@pytest.fixture
def dummy_trial():
    """Return a dummy trial object"""
    trial = Trial()
    trial._params = [
        Trial.Param(name="a", type="real", value=0.0),
        Trial.Param(name="b", type="integer", value=1),
        Trial.Param(name="c", type="categorical", value="Some"),
    ]
    return trial


@pytest.fixture
def dummy_dict():
    """Return a dict of dicts"""
    return {
        1: {1.1: "1.1.1", 1.2: {"1.2.1": {}, "1.2.2": "1.2.2.1"}},
        2: {2.1: "2.1.1", 2.2: {}},
        3: {},
    }


@pytest.fixture
def dummy_list_of_lists():
    """Return a list of lists"""
    return [1, [2, 3], 4, [5, 6, 7, 8]]


@pytest.fixture
def dummy_list_of_objects(dummy_dict):
    """Return a list of objects"""
    return [
        {
            1: {1.1: "1.1.1", 1.2: ["1.2.1", "1.2.2"]},
        },
        [4, 5],
        {2: {2.1: "2.1.1", 2.2: {}}, 3: {}},
    ]


@pytest.fixture
def algorithm_dict():
    """Return an algorithm configuration"""
    return dict(
        bayesianoptimizer=dict(
            acq_func="gp_hedge",
            alpha=1e-10,
            n_initial_points=10,
            n_restarts_optimizer=0,
            normalize_y=False,
        )
    )


def test_format_title():
    """Test title formatting template"""
    result = """Test\n===="""

    assert format_title("Test") == result


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_format_dict_depth_synthetic(depth, dummy_dict):
    """Test dict formatting with different depths for one line"""
    WIDTH = 4
    tab = (" " * WIDTH) * depth
    assert (
        format_dict(dummy_dict, depth=depth, width=WIDTH)
        .split("\n")[0]
        .startswith(tab + "1")
    )


def test_format_dict_depth_full(dummy_dict):
    """Test dict depth formatting for all lines"""
    WIDTH = 4
    tab = " " * WIDTH

    lines = format_dict(dummy_dict).split("\n")
    assert len(lines) == 9
    assert lines[0].startswith("1")
    assert lines[1].startswith(tab + "1")
    assert lines[2].startswith(tab + "1")
    assert lines[3].startswith((tab * 2) + "1")
    assert lines[4].startswith((tab * 2) + "1")
    assert lines[5].startswith("2")
    assert lines[6].startswith(tab + "2")
    assert lines[7].startswith(tab + "2")
    assert lines[8].startswith("3")


@pytest.mark.parametrize("width,depth", itertools.product([0, 2], [1, 2]))
def test_format_dict_width_synthetic(width, depth, dummy_dict):
    """Test dict formatting with different combination of widths and depth for one line"""
    tab = (" " * width) * depth
    assert (
        format_dict(dummy_dict, depth=depth, width=width)
        .split("\n")[0]
        .startswith(tab + "1")
    )


@pytest.mark.parametrize("width", [0, 5, 12])
def test_format_dict_width_full(width, dummy_dict):
    """Test dict formatting with different widths for all lines"""
    tab = " " * width
    lines = format_dict(dummy_dict, width=width).split("\n")
    assert len(lines) == 9
    assert lines[0].startswith("1")
    assert lines[1].startswith(tab + "1")
    assert lines[2].startswith(tab + "1")
    assert lines[3].startswith((tab * 2) + "1")
    assert lines[4].startswith((tab * 2) + "1")
    assert lines[5].startswith("2")
    assert lines[6].startswith(tab + "2")
    assert lines[7].startswith(tab + "2")
    assert lines[8].startswith("3")


def test_format_dict_empty_leaf_template():
    """Test dict empty leaf node formatting"""
    dummy_dict_with_leafs = {1: {2: {}}, 2: {}, 3: {1: 3, 2: {}}}
    lines = format_dict(dummy_dict_with_leafs).split("\n")
    assert len(lines) == 6
    # 1:
    assert lines[1].lstrip(" ") == "2"
    assert lines[2].lstrip(" ") == "2"
    # 3:
    #   1: 3
    assert lines[5].lstrip(" ") == "2"


def test_format_dict_empty_leaf_template_custom():
    """Test dict empty leaf node formatting with custom template"""
    template = "{key} is a leaf\n"
    dummy_dict_with_leafs = {1: {2: {}}, 2: {}, 3: {1: 3, 2: {}}}
    lines = format_dict(
        dummy_dict_with_leafs, templates={"empty_leaf": template}
    ).split("\n")
    assert len(lines) == 6
    # 1:
    assert lines[1] == "2 is a leaf"
    assert lines[2] == "2 is a leaf"
    # 3:
    #   1: 3
    assert lines[5] == "2 is a leaf"


def test_format_dict_leaf_template():
    """Test dict leaf node formatting"""
    dummy_dict_with_leafs = {1: {2: {}}, 2: {}, 3: {1: 3, 2: {}}}
    lines = format_dict(dummy_dict_with_leafs).split("\n")
    assert len(lines) == 6
    # 1:
    #   2
    # 2
    # 3:
    assert lines[4].lstrip(" ") == "1: 3"


def test_format_dict_leaf_template_custom():
    """Test dict leaf node formatting with custom template"""
    template = "value of {key} is {value}\n"
    dummy_dict_with_leafs = {1: {2: {}}, 2: {}, 3: {1: 3, 2: {}}}
    lines = format_dict(dummy_dict_with_leafs, templates={"leaf": template}).split("\n")
    assert len(lines) == 6
    # 1:
    #   2
    # 2
    # 3:
    assert lines[4] == "value of 1 is 3"


def test_format_dict_node_template():
    """Test dict node formatting"""
    dummy_dict_with_leafs = {1: {2: {}}, 2: {}, 3: {1: {4: 3}, 2: {}}}
    lines = format_dict(dummy_dict_with_leafs).split("\n")
    assert len(lines) == 7
    assert lines[0] == "1:"
    #   2
    # 2
    assert lines[3] == "3:"
    assert lines[4].lstrip(" ") == "1:"
    #     4: 3
    #   2


def test_format_dict_node_template_custom():
    """Test dict formatting with custom node template"""
    template = "{key} is a dict:\n{value}\n"
    dummy_dict_with_leafs = {1: {2: {}}, 2: {}, 3: {1: {4: 3}, 2: {}}}
    lines = format_dict(dummy_dict_with_leafs, templates={"dict_node": template}).split(
        "\n"
    )
    assert len(lines) == 7
    assert lines[0] == "1 is a dict:"
    #   2
    # 2
    assert lines[3] == "3 is a dict:"
    assert lines[4] == "1 is a dict:"
    #     4: 3
    #   2


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_format_list_depth_synthetic(depth, dummy_list_of_lists):
    """Test list of lists formatting with different depths for one line"""
    WIDTH = 4
    tab = (" " * WIDTH) * depth
    formatted_list_str = format_list(dummy_list_of_lists, depth=depth, width=WIDTH)
    assert formatted_list_str.split("\n")[0].startswith(tab + "[")


def test_format_list_depth_full(dummy_list_of_lists):
    """Test list of lists depth formatting for all lines"""
    WIDTH = 4
    tab = " " * WIDTH

    lines = format_list(dummy_list_of_lists).split("\n")
    assert len(lines) == 14
    assert lines[0] == "["
    assert lines[1] == tab + "1"
    assert lines[2] == tab + "["
    assert lines[3] == (tab * 2) + "2"
    assert lines[4] == (tab * 2) + "3"
    assert lines[5] == tab + "]"
    assert lines[6] == tab + "4"
    assert lines[7] == tab + "["
    assert lines[8] == (tab * 2) + "5"
    assert lines[9] == (tab * 2) + "6"
    assert lines[10] == (tab * 2) + "7"
    assert lines[11] == (tab * 2) + "8"
    assert lines[12] == tab + "]"
    assert lines[13] == "]"


@pytest.mark.parametrize("width,depth", itertools.product([0, 2], [1, 2]))
def test_format_list_width_synthetic(width, depth, dummy_list_of_lists):
    """Test list of lists formatting with different combination of widths and depth for one line"""
    tab = (" " * width) * depth
    assert (
        format_list(dummy_list_of_lists, depth=depth, width=width).split("\n")[0]
        == tab + "["
    )


@pytest.mark.parametrize("width", [0, 5, 12])
def test_format_list_width_full(width, dummy_list_of_lists):
    """Test list of lists formatting with different widths for all lines"""
    tab = " " * width
    lines = format_list(dummy_list_of_lists, width=width).split("\n")
    assert len(lines) == 14
    assert lines[0] == "["
    assert lines[1] == tab + "1"
    assert lines[2] == tab + "["
    assert lines[3] == (tab * 2) + "2"
    assert lines[4] == (tab * 2) + "3"
    assert lines[5] == tab + "]"
    assert lines[6] == tab + "4"
    assert lines[7] == tab + "["
    assert lines[8] == (tab * 2) + "5"
    assert lines[9] == (tab * 2) + "6"
    assert lines[10] == (tab * 2) + "7"
    assert lines[11] == (tab * 2) + "8"
    assert lines[12] == tab + "]"
    assert lines[13] == "]"


def test_format_list_item_template(dummy_list_of_lists):
    """Test list of lists formatting of items"""
    lines = format_list(dummy_list_of_lists).split("\n")
    assert len(lines) == 14
    assert lines[1].lstrip(" ") == "1"
    assert lines[3].lstrip(" ") == "2"


def test_format_list_item_template_custom(dummy_list_of_lists):
    """Test list of lists with custom item template"""
    template = "{item} is an item\n"
    lines = format_list(dummy_list_of_lists, templates={"item": template}).split("\n")
    assert len(lines) == 14
    # 1:
    assert lines[1] == "1 is an item"
    assert lines[3] == "2 is an item"


def test_format_list_node_template(dummy_list_of_lists):
    """Test list of lists formatting of node"""
    WIDTH = 4
    tab = " " * WIDTH
    lines = format_list(dummy_list_of_lists).split("\n")
    assert len(lines) == 14
    assert lines[2] == tab + "["
    assert lines[3] == (tab * 2) + "2"
    assert lines[4] == (tab * 2) + "3"
    assert lines[5] == tab + "]"


def test_format_list_node_template_custom(dummy_list_of_lists):
    """Test list of lists custom formatting"""
    templates = dict(list="[{items}]", item="{item}", list_node="{item}")
    lines = format_list(dummy_list_of_lists, templates=templates).split("\n")
    assert len(lines) == 1
    assert lines[0] == "[1[23]4[5678]]"


def test_format_dict_with_list(dummy_list_of_objects):
    """Test dict formatting with embedded lists"""
    assert (
        format_dict(dummy_list_of_objects)
        == """\
[
    1:
        1.1: 1.1.1
        1.2:
            [
                1.2.1
                1.2.2
            ]
    [
        4
        5
    ]
    2:
        2.1: 2.1.1
        2.2
    3
]\
"""
    )


def test_format_identification():
    """Test commandline section formatting"""
    experiment = DummyExperiment()
    experiment.name = "test"
    experiment.version = 1
    experiment.metadata = {"user": "corneauf"}
    assert (
        format_identification(experiment)
        == """\
Identification
==============
name: test
version: 1
user: corneauf
"""
    )


def test_format_commandline():
    """Test commandline section formatting"""
    experiment = DummyExperiment()
    commandline = ["executing.sh", "--some", "random", "--command", "line", "arguments"]
    experiment.metadata = {"user_args": commandline}
    assert (
        format_commandline(experiment)
        == """\
Commandline
===========
executing.sh --some random --command line arguments
"""
    )


def test_format_config(monkeypatch):
    """Test config section formatting"""
    experiment = DummyExperiment()
    experiment.pool_size = 10
    experiment.max_trials = 100
    experiment.max_broken = 5
    experiment.working_dir = "working_dir"
    assert (
        format_config(experiment)
        == """\
Config
======
pool size: 10
max trials: 100
max broken: 5
working dir: working_dir
"""
    )


def test_format_algorithm(algorithm_dict):
    """Test algorithm section formatting"""
    experiment = DummyExperiment()
    experiment.configuration = {"algorithms": algorithm_dict}
    assert (
        format_algorithm(experiment)
        == """\
Algorithm
=========
bayesianoptimizer:
    acq_func: gp_hedge
    alpha: 1e-10
    n_initial_points: 10
    n_restarts_optimizer: 0
    normalize_y: False
"""
    )


def test_format_space():
    """Test space section formatting"""
    experiment = DummyExperiment()
    space = SpaceBuilder().build(
        {"some": 'choices(["random", "or", "not"])', "command": "uniform(0, 1)"}
    )
    experiment.space = space
    assert (
        format_space(experiment)
        == """\
Space
=====
command: uniform(0, 1)
some: choices(['random', 'or', 'not'])
"""
    )


def test_format_metadata():
    """Test metadata section formatting"""
    experiment = DummyExperiment()
    experiment.metadata = dict(
        user="user",
        datetime="now",
        orion_version="1.0.1",
        VCS=dict(
            HEAD_sha="sha",
            active_branch="branch",
            diff_sha="smt",
            is_dirty=True,
            type="git",
        ),
    )
    assert (
        format_metadata(experiment)
        == """\
Meta-data
=========
user: user
datetime: now
orion version: 1.0.1
VCS:
  HEAD_sha: sha
  active_branch: branch
  diff_sha: smt
  is_dirty: True
  type: git
"""
    )


def test_format_refers_root():
    """Test refers section formatting for a root experiment"""
    experiment = DummyExperiment()
    experiment.node = DummyExperiment()
    experiment.node.root = experiment.node

    # experiment.refers = dict(
    #     parent='user',
    #     datetime='now',
    #     orion_version='1.0.1')
    assert (
        format_refers(experiment)
        == """\
Parent experiment
=================
root: 
parent: 
adapter: 
"""
    )  # noqa: W291


def test_format_refers_child():
    """Test refers section formatting for a child experiment"""
    ROOT_NAME = "root-name"
    PARENT_NAME = "parent-name"

    root = DummyExperiment()
    root.name = ROOT_NAME

    parent = DummyExperiment()
    parent.name = PARENT_NAME

    child = DummyExperiment()
    child.node = DummyExperiment()
    child.node.parent = parent
    child.node.root = root

    adapter = DummyExperiment()
    adapter.configuration = dict(adummy="dict", foran="adapter")

    child.refers = dict(adapter=adapter)

    # experiment.refers = dict(
    #     parent='user',
    #     datetime='now',
    #     orion_version='1.0.1')
    assert (
        format_refers(child)
        == """\
Parent experiment
=================
root: root-name
parent: parent-name
adapter: 
  adummy: dict
  foran: adapter
"""
    )  # noqa: W291


def test_get_trial_params_empty():
    """Test failing to fetch trials does not fail"""
    experiment = DummyExperiment()
    experiment.get_trial = lambda trial=None, uid=None: None
    assert get_trial_params(None, experiment) == {}


def test_get_trial_params(dummy_trial):
    """Test params are converted properly to a dict."""
    experiment = DummyExperiment()
    experiment.get_trial = lambda trial=None, uid=None: dummy_trial
    params = get_trial_params(None, experiment)
    assert params["a"] == 0.0
    assert params["b"] == 1
    assert params["c"] == "Some"


def test_format_stats(dummy_trial):
    """Test stats section formatting"""
    experiment = DummyExperiment()
    experiment.stats = dict(
        best_trials_id="dummy",
        trials_completed=10,
        best_evaluation=0.1,
        start_time="yesterday",
        finish_time="now",
        duration="way too long",
    )
    experiment.get_trial = lambda trial=None, uid=None: dummy_trial
    experiment.is_done = False
    assert (
        format_stats(experiment)
        == """\
Stats
=====
completed: False
trials completed: 10
best trial:
  id: dummy
  evaluation: 0.1
  params:
    a: 0.0
    b: 1
    c: Some
start time: yesterday
finish time: now
duration: way too long
"""
    )


def test_format_info(algorithm_dict, dummy_trial):
    """Test full formatting string"""
    experiment = DummyExperiment()
    commandline = [
        "executing.sh",
        '--some~choices(["random", "or", "not"])',
        "--command~uniform(0, 1)",
    ]
    experiment.name = "test"
    experiment.version = 1
    experiment.metadata = {"user_args": commandline}
    experiment.pool_size = 10
    experiment.max_trials = 100
    experiment.max_broken = 5
    experiment.working_dir = "working_dir"
    experiment.configuration = {"algorithms": algorithm_dict}

    space = SpaceBuilder().build(
        {"some": 'choices(["random", "or", "not"])', "command": "uniform(0, 1)"}
    )
    experiment.space = space
    experiment.metadata.update(
        dict(
            user="user",
            datetime="now",
            orion_version="1.0.1",
            VCS=dict(
                HEAD_sha="sha",
                active_branch="branch",
                diff_sha="smt",
                is_dirty=True,
                type="git",
            ),
        )
    )

    ROOT_NAME = "root-name"
    PARENT_NAME = "parent-name"

    root = DummyExperiment()
    root.name = ROOT_NAME

    parent = DummyExperiment()
    parent.name = PARENT_NAME

    experiment.node = DummyExperiment()
    experiment.node.parent = parent
    experiment.node.root = root

    adapter = DummyExperiment()
    adapter.configuration = dict(adummy="dict", foran="adapter")

    experiment.refers = dict(adapter=adapter)
    experiment.stats = dict(
        best_trials_id="dummy",
        trials_completed=10,
        best_evaluation=0.1,
        start_time="yesterday",
        finish_time="now",
        duration="way too long",
    )
    experiment.get_trial = lambda trial=None, uid=None: dummy_trial
    experiment.is_done = False

    assert (
        format_info(experiment)
        == """\
Identification
==============
name: test
version: 1
user: user


Commandline
===========
executing.sh --some~choices(["random", "or", "not"]) --command~uniform(0, 1)


Config
======
pool size: 10
max trials: 100
max broken: 5
working dir: working_dir


Algorithm
=========
bayesianoptimizer:
    acq_func: gp_hedge
    alpha: 1e-10
    n_initial_points: 10
    n_restarts_optimizer: 0
    normalize_y: False


Space
=====
command: uniform(0, 1)
some: choices(['random', 'or', 'not'])


Meta-data
=========
user: user
datetime: now
orion version: 1.0.1
VCS:
  HEAD_sha: sha
  active_branch: branch
  diff_sha: smt
  is_dirty: True
  type: git


Parent experiment
=================
root: root-name
parent: parent-name
adapter: 
  adummy: dict
  foran: adapter


Stats
=====
completed: False
trials completed: 10
best trial:
  id: dummy
  evaluation: 0.1
  params:
    a: 0.0
    b: 1
    c: Some
start time: yesterday
finish time: now
duration: way too long

"""
    )  # noqa: W291
