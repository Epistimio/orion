#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.io.evc_builder`."""


import pytest

from orion.core.io.evc_builder import (
    build_tree, build_trimmed_tree, fetch_nodes, trim_tree)
from orion.core.worker.experiment import ExperimentView


@pytest.mark.usefixtures("refers_id_substitution")
def test_fetch_nodes(create_db_instance):
    """Verify that all nodes from an EVC tree are fetched"""
    experiment = ExperimentView('supernaedo2.3')
    nodes = fetch_nodes(experiment.refers['root_id'])

    assert experiment.refers['root_id'] in [n['_id'] for n in nodes]
    assert len(nodes) == 9

    experiment = ExperimentView('supernaedo2.3.1.1')
    nodes = fetch_nodes(experiment.refers['root_id'])

    assert experiment.refers['root_id'] in [n['_id'] for n in nodes]
    assert len(nodes) == 9


@pytest.mark.usefixtures("refers_id_substitution")
def test_build_tree(create_db_instance):
    """Verify that built tree has the proper structure"""
    experiment = ExperimentView('supernaedo2.3')
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)

    assert root_node.item == 'supernaedo2'
    assert root_node.children[0].item == 'supernaedo2.1'
    assert root_node.children[1].item == 'supernaedo2.2'
    assert root_node.children[1].children[0].item == 'supernaedo2.2.1'
    assert root_node.children[2].item == 'supernaedo2.3'
    assert root_node.children[2].children[0].item == 'supernaedo2.3.1'
    assert root_node.children[2].children[0].children[0].item == 'supernaedo2.3.1.1'
    assert root_node.children[2].children[0].children[1].item == 'supernaedo2.3.1.2'
    assert root_node.children[2].children[0].children[2].item == 'supernaedo2.3.1.3'

    # Make sure there is no duplicate nodes
    assert len(list(root_node)) == 9


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_self_root_self_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping current node"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2.3'
    leaf_names = ['supernaedo2.3']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert len(list(exp_tree_node)) == 1


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_direct_root_self_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping parent"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.parent.item == root_name
    assert len(list(exp_tree_node.root)) == 2


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_deep_root_self_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping all parents"""
    experiment_name = 'supernaedo2.2.1'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.2.1']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.parent.parent.item == root_name
    assert len(list(exp_tree_node.root)) == 3


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_self_root_direct_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping child"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2.3'
    leaf_names = ['supernaedo2.3.1']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.children[0].item == leaf_names[0]
    assert len(list(exp_tree_node.root)) == 2


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_self_root_deep_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping path to a leaf"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2.3'
    leaf_names = ['supernaedo2.3.1.2']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.children[0].children[0].item == leaf_names[0]
    assert len(list(exp_tree_node.root)) == 3


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_self_root_multi_deep_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping path to multiple leafs"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2.3'
    leaf_names = ['supernaedo2.3.1.1', 'supernaedo2.3.1.2']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.children[0].children[0].item == leaf_names[0]
    assert exp_tree_node.children[0].children[1].item == leaf_names[1]
    assert len(list(exp_tree_node.root)) == 4


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_non_connected_root(create_db_instance):
    """Verify the function raises an error if specified root is not connected to experiment"""
    experiment_name = 'supernaedo2.2.1'
    root_name = 'supernaedo2.3'
    leaf_names = ['supernaedo2.2.1']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)

    with pytest.raises(RuntimeError) as exc_info:
        trim_tree(root_node, experiment_name, root_name, leaf_names)
    assert "Some experiments were not found: {'supernaedo2.3'}" in str(exc_info.value)


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_self_root_non_connected_leaf(create_db_instance):
    """Verify the function raises an error if specified leaf is not connected to experiment"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2.3'
    leaf_names = ['supernaedo2.3.1.1', 'supernaedo2.2.1']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)

    with pytest.raises(RuntimeError) as exc_info:
        trim_tree(root_node, experiment_name, root_name, leaf_names)
    assert "Some experiments were not found: {'supernaedo2.2.1'}" in str(exc_info.value)


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_self_root_multi_depth_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping path to multiple leafs (of different depth)"""
    experiment_name = 'supernaedo2'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3.1.1', 'supernaedo2.2.1']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.children[0].children[0].item == leaf_names[1]
    assert exp_tree_node.children[1].children[0].children[0].item == leaf_names[0]
    # 2
    # |      \
    # 2.2,   2.3
    # |       |
    # 2.2.1  2.3.1
    #         |
    #        2.3.1.1
    assert len(list(exp_tree_node.root)) == 6


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_direct_root_direct_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping parent and leaf"""
    experiment_name = 'supernaedo2.2'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.2.1']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.parent.item == root_name
    assert exp_tree_node.children[0].item == leaf_names[0]
    # 2
    # |
    # 2.2
    # |
    # 2.2.1
    assert len(list(exp_tree_node.root)) == 3


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_direct_root_deep_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping parent and path to leaf"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3.1.2']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.parent.item == root_name
    assert exp_tree_node.children[0].children[0].item == leaf_names[0]
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    # |
    # 2.3.1.2
    assert len(list(exp_tree_node.root)) == 4


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_deep_root_direct_leaf(create_db_instance):
    """Verify that the tree is trimmed only keeping path to root and leaf"""
    experiment_name = 'supernaedo2.3.1'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3.1.1']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.parent.parent.item == root_name
    assert exp_tree_node.children[0].item == leaf_names[0]
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    # |
    # 2.3.1.1
    assert len(list(exp_tree_node.root)) == 4


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_no_root(create_db_instance):
    """Verify that the tree is not trimmed at root"""
    experiment_name = 'supernaedo2.3'
    root_name = None
    leaf_names = ['supernaedo2.3.1.1']

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.parent.item == 'supernaedo2'
    assert exp_tree_node.children[0].children[0].item == leaf_names[0]
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    # |
    # 2.3.1.1
    assert len(list(exp_tree_node.root)) == 4


@pytest.mark.usefixtures("refers_id_substitution")
def test_trim_tree_no_leaf(create_db_instance):
    """Verify that the tree is not trimmed at leafs"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2'
    leaf_names = []

    experiment = ExperimentView(experiment_name)
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment_name, root_name, leaf_names)

    assert exp_tree_node.item == experiment_name
    assert exp_tree_node.parent.item == root_name
    assert exp_tree_node.children[0].children[0].item == 'supernaedo2.3.1.1'
    assert exp_tree_node.children[0].children[1].item == 'supernaedo2.3.1.2'
    assert exp_tree_node.children[0].children[2].item == 'supernaedo2.3.1.3'
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    # |        \         \
    # 2.3.1.1  2.3.1.2  2.3.1.3
    assert len(list(exp_tree_node.root)) == 6


@pytest.mark.usefixtures("refers_id_substitution")
def test_build_trimmed_tree(create_db_instance):
    """Test trimmed tree converted to experiment nodes"""
    experiment_name = 'supernaedo2.3'
    root_name = None
    leaf_names = []

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == 'supernaedo2'
    assert exp_node.children[0].children[0].item.name == 'supernaedo2.3.1.1'
    assert exp_node.children[0].children[1].item.name == 'supernaedo2.3.1.2'
    assert exp_node.children[0].children[2].item.name == 'supernaedo2.3.1.3'
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    # |        \         \
    # 2.3.1.1  2.3.1.2  2.3.1.3
    assert len(list(exp_node.root)) == 6


@pytest.mark.usefixtures("refers_id_substitution")
def test_build_trimmed_tree_from_children_children(create_db_instance):
    """Test trimmed tree converted to experiment nodes with children experiment"""
    experiment_name = 'supernaedo2.3.1'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.parent.item.name == root_name
    assert len(exp_node.children) == 0
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    assert len(list(exp_node.root)) == 3
