# -*- coding: utf-8 -*-
# pylint:disable=protected-access
"""
:mod:`orion.core.evc.experiment` -- Experiment node for EVC
===========================================================

.. module:: experiment
   :platform: Unix
   :synopsis: Experiment nodes connecting experiments to the EVC tree

The experiments are connected to one another through the experiment nodes. The former can be created
standalone without an EVC tree. When connected to an `ExperimentNode`, the experiments gain access
to trials of other experiments by using method `ExperimentNode.fetch_trials`.

Helper functions are provided to fetch trials keeping the tree structure. Those can be helpful when
analyzing an EVC tree.

"""
import functools
import logging

from orion.core.evc.tree import TreeNode
from orion.core.io.database.base import Database
from orion.core.worker.experiment import ExperimentView

log = logging.getLogger(__name__)


class ExperimentNode(TreeNode):
    """Experiment node to connect experiments to EVC tree.

    The node carries an experiment in attribute `item`. The node can be instantiated only using the
    name of the experiment. The experiment will be created lazily on access to `node.item`.

    Attributes
    ----------
    name: str
        Name of the experiment
    item: None or :class:`orion.core.worker.experiment.Experiment`
        None if the experiment is not initialized yet. When initializing lazily, it creates an
        `ExperimentView`.

    .. seealso::

        :py:class:`TreeNode` for tree-specific attributes and methods.

    """

    __slots__ = ('name', '_no_parent_lookup', '_no_children_lookup') + TreeNode.__slots__

    def __init__(self, name, experiment=None, parent=None, children=tuple()):
        """Initialize experiment node with item, experiment, parent and children

        .. seealso::
            :class:`orion.core.evc.tree.TreeNode` for information about the attributes
        """
        super(ExperimentNode, self).__init__(experiment, parent, children)
        self.name = name
        self._no_parent_lookup = True
        self._no_children_lookup = True

    @property
    def item(self):
        """Get the experiment associated to the node

        Note that accessing `item` may trigger the lazy initialization of the experiment if it was
        not done already.
        """
        if self._item is None:
            self._item = ExperimentView(self.name)
            self._item.connect_to_version_control_tree(self)

        return self._item

    @property
    def parent(self):
        """Get parent of the experiment, None if no parent

        .. note::

            The instantiation of an EVC tree is lazy, which means accessing the parent of a node
            may trigger a call to database to build this parent live.

        """
        if self._parent is None and self._no_parent_lookup:
            self._no_parent_lookup = False
            query = {'_id': self.item.refers['parent_id']}
            selection = {'name': 1}
            experiments = Database().read('experiments', query, selection=selection)
            if experiments:
                self.set_parent(ExperimentNode(name=experiments[0]['name']))

        return self._parent

    @property
    def children(self):
        """Get children of the experiment, empty list if no children

        .. note::

            The instantiation of an EVC tree is lazy, which means accessing the children of a node
            may trigger a call to database to build those children live.

        """
        if not self._children and self._no_children_lookup:
            self._no_children_lookup = False
            query = {'refers.parent_id': self.item.id}
            selection = {'name': 1}
            experiments = Database().read('experiments', query, selection=selection)
            for child in experiments:
                self.add_children(ExperimentNode(name=child['name']))

        return self._children

    @property
    def adapter(self):
        """Get the adapter of the experiment with respect to its parent"""
        return self.item.refers["adapter"]

    def fetch_trials(self, query, selection=None):
        """Fetch trials recursively in the EVC tree

        .. seealso::

            :meth:`orion.core.worker.Experiment.fetch_trials` for more information about the
            arguments.

        """
        trials_tree = fetch_trials_tree(self, query, selection)
        adapt_trials(trials_tree)

        return sum([node.item['trials'] for node in trials_tree.root], [])


def _fetch_node_trials(experiment_node, parent_or_children, query, selection=None):
    """Fetch trials from the current node and connect with parent or children

    .. note::

        To call with node.map to connect with parents or children

    """
    experiment_trials = experiment_node.item.fetch_trials(query, selection)

    rval = {'trials': experiment_trials, 'experiment': experiment_node.item}

    return rval, parent_or_children


def fetch_trials_tree(experiment_node, query, selection=None):
    """Fetch trials recursively from an experiment node

    .. seealso::

        :meth:`orion.core.evc.experiment.ExperimentNode.fetch_trials`

    """
    if experiment_node.parent is not None:
        parent_trials_tree = experiment_node.parent.map(
            functools.partial(_fetch_node_trials, query=query, selection=selection),
            experiment_node.parent.parent)
    else:
        parent_trials_tree = None

    children_trials_tree = experiment_node.map(
        functools.partial(_fetch_node_trials, query=query, selection=selection),
        experiment_node.children)
    children_trials_tree.set_parent(parent_trials_tree)

    return children_trials_tree


def _adapt_parent_trials(node, parent_trials_node):
    """Adapt trials from the parent recursively

    .. note::

        To call with node.map(fct, node.parent) to connect with parents

    """
    if parent_trials_node is not None:
        adapter = node.item['experiment'].refers['adapter']
        for parent in parent_trials_node.root:
            parent.item['trials'] = adapter.forward(parent.item['trials'])

    return node.item, parent_trials_node


def _adapt_children_trials(node, children_trials_nodes):
    """Adapt trials from the children recursively

    .. note::

        To call with node.map(fct, node.children) to connect with children

    """
    for child in children_trials_nodes:
        adapter = child.item['experiment'].refers['adapter']
        for subchild in child:  # Includes child itself
            subchild.item['trials'] = adapter.backward(subchild.item['trials'])

    return node.item, children_trials_nodes


def adapt_trials(trials_tree):
    """Adapt trials recursively so that they are all compatible with current experiment."""
    trials_tree.map(_adapt_parent_trials, trials_tree.parent)
    trials_tree.map(_adapt_children_trials, trials_tree.children)
