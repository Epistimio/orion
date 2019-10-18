#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.evc.experiment`."""

import pytest

from orion.core.evc.adapters import Adapter, CodeChange
from orion.core.evc.experiment import ExperimentNode
from orion.core.worker.experiment import ExperimentView


# To avoid flake8 issues
build_trimmed_tree = "dummy"


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_parent_fetch_trials(create_db_instance):
    """Test that experiment fetch trials from parent properly (adapters are muted)"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3']

    experiment = ExperimentView(experiment_name)
    exp_node = ExperimentNode(experiment.name, experiment.version, experiment=experiment)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == root_name
    assert len(exp_node.children) == 0
    # 2
    # |
    # 2.3
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    for node in exp_node.root:
        node.item._experiment.refers['adapter'] = Adapter.build([])

    query = {'status': 'completed'}
    assert len(experiment.fetch_trials(query)) == 4
    assert len(experiment._experiment._node.parent.item.fetch_trials(query)) == 6
    assert len(experiment.fetch_trials_tree(query)) == 10


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_children_fetch_trials(create_db_instance):
    """Test that experiment fetch trials from children properly (adapters are muted)"""
    experiment_name = 'supernaedo2'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == leaf_names[0]
    assert len(exp_node.children) == 1
    # 2
    # |
    # 2.3
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    for node in exp_node.root:
        node.item._experiment.refers['adapter'] = Adapter.build([])

    query = {'status': 'completed'}
    assert len(experiment.fetch_trials(query)) == 6
    assert len(experiment._experiment._node.children[0].item.fetch_trials(query)) == 4
    assert len(experiment.fetch_trials_tree(query)) == 10


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_parent_parent_fetch_trials(create_db_instance):
    """Test that experiment fetch trials from grand parent properly (adapters are muted)"""
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

    experiment.connect_to_version_control_tree(exp_node)

    for node in exp_node.root:
        node.item._experiment.refers['adapter'] = Adapter.build([])

    query = {'status': 'completed'}
    assert len(exp_node.parent.parent.item.fetch_trials(query)) == 6
    assert len(exp_node.parent.item.fetch_trials(query)) == 4
    assert len(exp_node.item.fetch_trials(query)) == 2

    assert len(experiment.fetch_trials_tree(query)) == 6 + 4 + 2


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_children_children_fetch_trials(create_db_instance):
    """Test that experiment fetch trials from grand children properly (adapters are muted)"""
    experiment_name = 'supernaedo2'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].children[0].item.name == leaf_names[0]
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    assert len(list(exp_node.root)) == 3

    experiment.connect_to_version_control_tree(exp_node)

    for node in exp_node.root:
        node.item._experiment.refers['adapter'] = Adapter.build([])

    query = {'status': 'completed'}
    assert len(exp_node.item.fetch_trials(query)) == 6
    assert len(exp_node.children[0].item.fetch_trials(query)) == 4
    assert len(exp_node.children[0].children[0].item.fetch_trials(query)) == 2

    assert len(experiment.fetch_trials_tree(query)) == 6 + 4 + 2


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_deletion_adapter_forward(create_db_instance):
    """Test that all decoding_layer=gru pass to children"""
    experiment_name = 'supernaedo2.1'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == root_name
    assert len(exp_node.children) == 0
    # 2
    # |
    # 2.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    assert len(exp_node.item.fetch_trials(query)) == 1
    assert len(exp_node.parent.item.fetch_trials(query)) == 6

    adapter = experiment.refers['adapter']
    assert len(adapter.forward(exp_node.parent.item.fetch_trials(query))) == 1
    assert len(experiment.fetch_trials_tree(query)) == 1 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_deletion_adapter_backward(create_db_instance):
    """Test that all decoding_layer are passed with gru to parent"""
    experiment_name = 'supernaedo2'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == leaf_names[0]
    # 2
    # |
    # 2.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    assert len(exp_node.item.fetch_trials(query)) == 6
    assert len(exp_node.children[0].item.fetch_trials(query)) == 1

    adapter = exp_node.children[0].item.refers['adapter']
    assert len(adapter.backward(exp_node.children[0].item.fetch_trials(query))) == 1
    assert len(experiment.fetch_trials_tree(query)) == 6 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_renaming_forward(create_db_instance):
    """Test that all encoding_layer are renamed to encoding in children"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == root_name
    # 2
    # |
    # 2.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    parent_trials = exp_node.parent.item.fetch_trials(query)
    assert len(parent_trials) == 6
    assert len(exp_node.item.fetch_trials(query)) == 4

    assert all((trial._params[0].name == "/encoding_layer") for trial in parent_trials)

    adapter = experiment.refers['adapter']
    adapted_parent_trials = adapter.forward(parent_trials)
    assert len(adapted_parent_trials) == 6
    assert all((trial._params[0].name == "/encoding") for trial in adapted_parent_trials)

    assert len(experiment.fetch_trials_tree(query)) == 6 + 4


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_renaming_backward(create_db_instance):
    """Test that all encoding are renamed to encoding_layer in parent"""
    experiment_name = 'supernaedo2'
    root_name = 'supernaedo2'
    leaf_names = ['supernaedo2.3']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == leaf_names[0]
    # 2
    # |
    # 2.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    children_trials = exp_node.children[0].item.fetch_trials(query)
    assert len(children_trials) == 4
    assert len(exp_node.item.fetch_trials(query)) == 6

    assert all((trial._params[0].name == "/encoding") for trial in children_trials)

    adapter = exp_node.children[0].item.refers['adapter']
    adapted_children_trials = adapter.backward(children_trials)
    assert len(adapted_children_trials) == 4
    assert all((trial._params[0].name == "/encoding_layer") for trial in adapted_children_trials)

    assert len(experiment.fetch_trials_tree(query)) == 6 + 4


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_prior_change_forward(create_db_instance):
    """Test that trials from parent only pass to children if valid in the new prior"""
    experiment_name = 'supernaedo2.3.1'
    root_name = 'supernaedo2.3'
    leaf_names = ['supernaedo2.3.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == root_name
    # 2.3
    # |
    # 2.3.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    parent_trials = exp_node.parent.item.fetch_trials(query)
    assert len(parent_trials) == 4
    assert len(exp_node.item.fetch_trials(query)) == 2

    adapter = experiment.refers['adapter']
    adapted_parent_trials = adapter.forward(parent_trials)
    assert len(adapted_parent_trials) == 1
    assert len(experiment.fetch_trials_tree(query)) == 2 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_prior_change_backward(create_db_instance):
    """Test that all encoding are renamed to encoding_layer in parent"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2.3'
    leaf_names = ['supernaedo2.3.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == leaf_names[0]
    # 2.3
    # |
    # 2.3.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    children_trials = exp_node.children[0].item.fetch_trials(query)
    assert len(children_trials) == 2
    assert len(exp_node.item.fetch_trials(query)) == 4

    adapter = exp_node.children[0].item.refers['adapter']
    adapted_children_trials = adapter.backward(children_trials)
    assert len(adapted_children_trials) == 1

    assert len(experiment.fetch_trials_tree(query)) == 4 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_code_change_noeffect_forward(create_db_instance):
    """Test that all trials pass to children when code change type is 'noeffect'"""
    experiment_name = 'supernaedo2.3.1.1'
    root_name = 'supernaedo2.3.1'
    leaf_names = ['supernaedo2.3.1.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == root_name
    # 2.3.1
    # |
    # 2.3.1.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    parent_trials = exp_node.parent.item.fetch_trials(query)
    assert len(parent_trials) == 2
    assert len(exp_node.item.fetch_trials(query)) == 1

    adapter = experiment.refers['adapter']
    assert adapter.adapters[0].change_type == CodeChange.NOEFFECT
    adapted_parent_trials = adapter.forward(parent_trials)
    assert len(adapted_parent_trials) == 2
    assert len(experiment.fetch_trials_tree(query)) == 2 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_code_change_noeffect_backward(create_db_instance):
    """Test that all trials pass to parent when code change type is 'noeffect'"""
    experiment_name = 'supernaedo2.3.1'
    root_name = 'supernaedo2.3.1'
    leaf_names = ['supernaedo2.3.1.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == leaf_names[0]
    # 2.3
    # |
    # 2.3.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    children_trials = exp_node.children[0].item.fetch_trials(query)
    assert len(children_trials) == 1
    assert len(exp_node.item.fetch_trials(query)) == 2

    adapter = exp_node.children[0].item.refers['adapter']
    assert adapter.adapters[0].change_type == CodeChange.NOEFFECT
    adapted_children_trials = adapter.backward(children_trials)
    assert len(adapted_children_trials) == 1

    assert len(experiment.fetch_trials_tree(query)) == 2 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_code_change_unsure_forward(create_db_instance):
    """Test that all trials pass to children when code change type is 'unsure'"""
    experiment_name = 'supernaedo2.3.1.2'
    root_name = 'supernaedo2.3.1'
    leaf_names = ['supernaedo2.3.1.2']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == root_name
    # 2.3.1
    # |
    # 2.3.1.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    parent_trials = exp_node.parent.item.fetch_trials(query)
    assert len(parent_trials) == 2
    assert len(exp_node.item.fetch_trials(query)) == 1

    adapter = experiment.refers['adapter']
    assert adapter.adapters[0].change_type == CodeChange.UNSURE
    adapted_parent_trials = adapter.forward(parent_trials)
    assert len(adapted_parent_trials) == 2
    assert len(experiment.fetch_trials_tree(query)) == 2 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_code_change_unsure_backward(create_db_instance):
    """Test that no trials pass to parent when code change type is 'unsure'"""
    experiment_name = 'supernaedo2.3.1'
    root_name = 'supernaedo2.3.1'
    leaf_names = ['supernaedo2.3.1.2']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == leaf_names[0]
    # 2.3
    # |
    # 2.3.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    children_trials = exp_node.children[0].item.fetch_trials(query)
    assert len(children_trials) == 1
    assert len(exp_node.item.fetch_trials(query)) == 2

    adapter = exp_node.children[0].item.refers['adapter']
    assert adapter.adapters[0].change_type == CodeChange.UNSURE
    adapted_children_trials = adapter.backward(children_trials)
    assert len(adapted_children_trials) == 0

    assert len(experiment.fetch_trials_tree(query)) == 2 + 0


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_code_change_break_forward(create_db_instance):
    """Test that no trials pass to children when code change type is 'break'"""
    experiment_name = 'supernaedo2.3.1.3'
    root_name = 'supernaedo2.3.1'
    leaf_names = ['supernaedo2.3.1.3']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == root_name
    # 2.3.1
    # |
    # 2.3.1.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    parent_trials = exp_node.parent.item.fetch_trials(query)
    assert len(parent_trials) == 2
    assert len(exp_node.item.fetch_trials(query)) == 1

    adapter = experiment.refers['adapter']
    assert adapter.adapters[0].change_type == CodeChange.BREAK
    adapted_parent_trials = adapter.forward(parent_trials)
    assert len(adapted_parent_trials) == 0
    assert len(experiment.fetch_trials_tree(query)) == 1 + 0


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_code_change_break_backward(create_db_instance):
    """Test that no trials pass to parent when code change type is 'break'"""
    experiment_name = 'supernaedo2.3.1'
    root_name = 'supernaedo2.3.1'
    leaf_names = ['supernaedo2.3.1.3']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == leaf_names[0]
    # 2.3
    # |
    # 2.3.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    children_trials = exp_node.children[0].item.fetch_trials(query)
    assert len(children_trials) == 1
    assert len(exp_node.item.fetch_trials(query)) == 2

    adapter = exp_node.children[0].item.refers['adapter']
    assert adapter.adapters[0].change_type == CodeChange.BREAK
    adapted_children_trials = adapter.backward(children_trials)
    assert len(adapted_children_trials) == 0

    assert len(experiment.fetch_trials_tree(query)) == 2 + 0


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_algo_change_forward(create_db_instance):
    """Test that all trials pass to children when algorithm is changed"""
    experiment_name = 'supernaedo2.2.1'
    root_name = 'supernaedo2.2'
    leaf_names = ['supernaedo2.2.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == root_name
    # 2.2
    # |
    # 2.2.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    parent_trials = exp_node.parent.item.fetch_trials(query)
    assert len(parent_trials) == 1
    assert len(exp_node.item.fetch_trials(query)) == 1

    adapter = experiment.refers['adapter']
    adapted_parent_trials = adapter.forward(parent_trials)
    assert len(adapted_parent_trials) == 1
    assert len(experiment.fetch_trials_tree(query)) == 1 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_algo_change_backward(create_db_instance):
    """Test that all trials pass to parent when algorithm is changed"""
    experiment_name = 'supernaedo2.2'
    root_name = 'supernaedo2.2'
    leaf_names = ['supernaedo2.2.1']

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == leaf_names[0]
    # 2.2
    # |
    # 2.2.1
    assert len(list(exp_node.root)) == 2

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    children_trials = exp_node.children[0].item.fetch_trials(query)
    assert len(children_trials) == 1
    assert len(exp_node.item.fetch_trials(query)) == 1

    adapter = exp_node.children[0].item.refers['adapter']
    adapted_children_trials = adapter.backward(children_trials)
    assert len(adapted_children_trials) == 1

    assert len(experiment.fetch_trials_tree(query)) == 1 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_full_backward(create_db_instance):
    """Test that trials are adapted properly up to root"""
    experiment_name = 'supernaedo2'
    root_name = None
    leaf_names = []

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.children[0].item.name == 'supernaedo2.1'
    assert exp_node.children[1].item.name == 'supernaedo2.2'
    assert exp_node.children[2].item.name == 'supernaedo2.3'
    assert exp_node.children[1].children[0].item.name == 'supernaedo2.2.1'
    assert exp_node.children[2].children[0].item.name == 'supernaedo2.3.1'
    assert exp_node.children[2].children[0].children[0].item.name == 'supernaedo2.3.1.1'
    assert exp_node.children[2].children[0].children[1].item.name == 'supernaedo2.3.1.2'
    assert exp_node.children[2].children[0].children[2].item.name == 'supernaedo2.3.1.3'
    # 2
    # |    \      \
    # 2.1  2.2    2.3
    #      |      |
    #      2.2.1  2.3.1
    #             |        \         \
    #             2.3.1.1  2.3.1.2  2.3.1.3
    assert len(list(exp_node)) == 9

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    assert len(exp_node.item.fetch_trials(query)) == 6
    assert len(exp_node.children[0].item.fetch_trials(query)) == 1
    assert len(exp_node.children[1].item.fetch_trials(query)) == 1
    assert len(exp_node.children[2].item.fetch_trials(query)) == 4
    assert len(exp_node.children[1].children[0].item.fetch_trials(query)) == 1
    assert len(exp_node.children[2].children[0].item.fetch_trials(query)) == 2
    assert len(exp_node.children[2].children[0].children[0].item.fetch_trials(query)) == 1
    assert len(exp_node.children[2].children[0].children[1].item.fetch_trials(query)) == 1
    assert len(exp_node.children[2].children[0].children[2].item.fetch_trials(query)) == 1

    assert len(experiment.fetch_trials_tree(query)) == 15


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_full_forward(create_db_instance):
    """Test that trials are adapted properly down to leaf"""
    experiment_name = 'supernaedo2.3.1.1'
    root_name = None
    leaf_names = []

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.item.name == 'supernaedo2.3.1'
    assert exp_node.parent.parent.item.name == 'supernaedo2.3'
    assert exp_node.parent.parent.parent.item.name == 'supernaedo2'
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    # |
    # 2.3.1.1
    assert len(list(exp_node.root)) == 4

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    assert len(exp_node.item.fetch_trials(query)) == 1
    assert len(exp_node.parent.item.fetch_trials(query)) == 2
    assert len(exp_node.parent.parent.item.fetch_trials(query)) == 4
    assert len(exp_node.parent.parent.parent.item.fetch_trials(query)) == 6

    assert len(experiment.fetch_trials_tree(query)) == 2 + 1 + 2 + 1


@pytest.mark.skip(reason="Support for leafs and parents dropped for now")
@pytest.mark.usefixtures("trial_id_substitution")
@pytest.mark.usefixtures("refers_id_substitution")
def test_full_forward_full_backward(create_db_instance):
    """Test that trials are adapted properly forward from parent and backward from leafs"""
    experiment_name = 'supernaedo2.3'
    root_name = 'supernaedo2'
    leaf_names = []

    experiment = ExperimentView(experiment_name)
    exp_node = build_trimmed_tree(experiment, root_name, leaf_names)

    assert exp_node.item.name == experiment_name
    assert exp_node.parent.name == root_name
    assert exp_node.children[0].item.name == 'supernaedo2.3.1'
    assert exp_node.children[0].children[0].item.name == 'supernaedo2.3.1.1'
    assert exp_node.children[0].children[1].item.name == 'supernaedo2.3.1.2'
    assert exp_node.children[0].children[2].item.name == 'supernaedo2.3.1.3'
    # 2
    # |
    # 2.3
    # |
    # 2.3.1
    # |        \        \
    # 2.3.1.1  2.3.1.2  2.3.1.3
    assert len(list(exp_node.root)) == 6

    experiment.connect_to_version_control_tree(exp_node)

    query = {'status': 'completed'}
    assert len(exp_node.parent.item.fetch_trials(query)) == 6
    assert len(exp_node.item.fetch_trials(query)) == 4
    assert len(exp_node.children[0].item.fetch_trials(query)) == 2
    assert len(exp_node.children[0].children[0].item.fetch_trials(query)) == 1
    assert len(exp_node.children[0].children[1].item.fetch_trials(query)) == 1
    assert len(exp_node.children[0].children[2].item.fetch_trials(query)) == 1

    assert len(experiment.fetch_trials_tree(query)) == 6 + 4 + 1 + 1 + 0 + 0
