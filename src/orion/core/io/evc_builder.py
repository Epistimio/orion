# -*- coding: utf-8 -*-
# pylint:disable=protected-access
"""
:mod:`orion.core.io.evc_builder` -- Builder of experiment version control tree
==============================================================================

.. module:: experiment
   :platform: Unix
   :synopsis: Builder of the experiment version control tree

The EVCBuilder takes care of building a main experiment along with an EVC tree and connect them
together.

A user can define a root and some leafs that should be the extremums of the tree. Those can be
different than the actual root and leafs of the global EVC tree, making the trimmed version a small
subset of the global version.

"""
import getpass

from orion.core.evc.experiment import ExperimentNode
from orion.core.evc.tree import TreeNode
from orion.core.io.database import Database
from orion.core.io.experiment_builder import ExperimentBuilder


class EVCBuilder(object):
    """Builder of experiment version control trees using
    :class:`orion.core.evc.experiment.ExperimentNode`

    .. seealso::

        `orion.core.io.experiment_builder` for more information on the process of building
        experiments.

        :class:`orion.core.evc.experiment`
        :class:`orion.core.worker.experiment`
    """

    # pylint:disable=no-self-use
    def fetch_root_name(self, cmdargs):
        """Fetch root name to trim EVC tree"""
        return cmdargs.get('root', None)

    # pylint:disable=no-self-use
    def fetch_leaf_names(self, cmdargs):
        """Fetch leaf names to trim EVC tree"""
        return cmdargs.get('leafs', None)

    def build_trimmed_tree(self, experiment, cmdargs):
        """Build a tree of `ExperimentNode` trimmed according to `root_name` and `leaf_names` found
        in configuration
        """
        root_name = self.fetch_root_name(cmdargs)
        leaf_names = self.fetch_leaf_names(cmdargs)

        return build_trimmed_tree(experiment, root_name, leaf_names)

    def connect_to_version_control_tree(self, experiment, cmdargs):
        """Build the EVC and connect the experiment to it"""
        experiment_node = self.build_trimmed_tree(experiment, cmdargs)
        experiment.connect_to_version_control_tree(experiment_node)

    def build_view_from(self, cmdargs):
        """Build an experiment view based on global config and connect it to the EVC"""
        experiment_view = ExperimentBuilder().build_view_from(cmdargs)
        self.connect_to_version_control_tree(experiment_view, cmdargs)

        return experiment_view

    def build_from(self, cmdargs):
        """Build an experiment based on config and connect it to the EVC"""
        experiment = ExperimentBuilder().build_from(cmdargs)
        self.connect_to_version_control_tree(experiment, cmdargs)

        return experiment

    def build_from_config(self, config):
        """Build an experiment based on given config and connect it to the EVC"""
        experiment = ExperimentBuilder().build_from_config(config)
        self.connect_to_version_control_tree(experiment, config)

        return experiment


def fetch_nodes(experiment_id):
    """Query all experiment documents which matches the root_id"""
    query = {'refers.root_id': experiment_id, "metadata.user": getpass.getuser()}
    selection = {'name': 1, 'refers.root_id': 1, 'refers.parent_id': 1}
    nodes = Database().read('experiments', query, selection=selection)
    return nodes


def _fetch_root(root_id, free_nodes):
    """Find the root in the set of nodes"""
    root = None

    for node in list(free_nodes):
        if node['_id'] == root_id:
            root = free_nodes.pop(free_nodes.index(node))
            break

    if root is None:
        raise ValueError("Found no experiment with id '%s'" % str(root_id))

    return root


def build_tree(root_id, nodes, root=None):
    """Find the root in the set of nodes"""
    free_nodes = nodes

    if root is None:
        root = _fetch_root(root_id, free_nodes)

    children = []
    for node in nodes[:]:
        if node['refers']['parent_id'] == root_id:
            children.append(free_nodes.pop(free_nodes.index(node)))

    root = TreeNode(root['name'])

    for child in children:
        root.add_children(build_tree(child['_id'], free_nodes, child))

    return root


def trim_tree(root_node, experiment_name, root_name=None, leaf_names=tuple()):
    """Trim the tree by removing nodes above a given root name and below given leaf names.

    If `root_name` is `None`, the nothing will be trimmed at the root. If `leaf_names` is an empty
    iterator, nothing will be trimmed the at the leafs.

    Parameters
    ----------
    root_node: `orion.core.evc.tree.TreeNode`
        The root node of the tree to trim.
    experiment_name: str
        Name of the current experiment. May not be the root.
    root_name: str
        Name of the root at which the function should trim. If `None`, the tree won't be trimmed at
        its root. Defaults to `None`.
    leaf_names: tuple or list of str
        Name of the leafs at which the function should trim. Only the paths leading to the specified
        leafs will be kept.

    Raises
    ------
    RuntimeError
        Raise if the given `root_name` of the given `leaf_names` are not found in the tree.

    """
    if root_name is None:
        root_name = root_node.item

    experiment_node = None

    for node in list(root_node):
        if node.item == experiment_name:
            experiment_node = node
            break

    def trim_parent(node, parent):
        """Remove the parent if not on the path to specified root"""
        if parent is not None and parent.root.item != root_name:
            return node.item, None

        return node.item, parent

    trimmed_parent = experiment_node.map(trim_parent, experiment_node.parent).parent
    if trimmed_parent is not None:
        trimmed_parent.drop_children()
    experiment_node.set_parent(trimmed_parent)

    def trim_children(node, children):
        """Remove the children if not on the path to specified leafs"""
        trimmed_children = []
        for child in list(children):
            if child.children or child.item in leaf_names:
                trimmed_children.append(child)

        return node.item, trimmed_children

    if leaf_names:
        trimmed_children = experiment_node.map(trim_children, experiment_node.children)
        experiment_node.drop_children()
        experiment_node.add_children(*trimmed_children.children)

    unseen = set([root_name] + leaf_names)

    for node in experiment_node.root:
        if node.item in unseen:
            unseen.remove(node.item)

    if unseen:
        raise RuntimeError("Some experiments were not found: %s" % str(unseen))

    return experiment_node


def convert_to_experiment_nodes(experiment_node):
    """Convert shallow `TreeNode` into initializable `ExperimentNode`"""
    def _convert_children_to_experiment_nodes(node, children):
        """Convert the children to `ExperimentNode`"""
        converted_children = []
        for child in children:
            converted_children.append(ExperimentNode(child.item, children=child.children))

        return node.item, converted_children

    def _convert_parent_to_experiment_nodes(node, parent):
        """Convert the parent to `ExperimentNode`"""
        if parent is None:
            return node.item, None

        grand_parent = parent.parent
        if grand_parent is not None:
            grand_parent.drop_children()

        return node.item, ExperimentNode(parent.item, parent=grand_parent)

    parent = experiment_node.parent
    if parent is not None:
        parent = experiment_node.map(_convert_parent_to_experiment_nodes,
                                     experiment_node.parent).parent
        parent.drop_children()

    children_nodes = experiment_node.map(_convert_children_to_experiment_nodes,
                                         experiment_node.children).children

    return ExperimentNode(experiment_node.item, parent=parent, children=children_nodes)


def build_trimmed_tree(experiment, root_name, leaf_names):
    """Build a trimmed tree for the EVC of a given experiment

    .. seealso::
        `orion.core.io.evc_builder.trim_tree` for more information about the arguments.

    """
    nodes = fetch_nodes(experiment.refers['root_id'])
    root_node = build_tree(experiment.refers['root_id'], nodes)
    exp_tree_node = trim_tree(root_node, experiment.name, root_name, leaf_names)

    return convert_to_experiment_nodes(exp_tree_node)
