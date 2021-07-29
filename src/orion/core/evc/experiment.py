# -*- coding: utf-8 -*-
# pylint:disable=protected-access
"""
Experiment node for EVC
=======================

Experiment nodes connecting experiments to the EVC tree

The experiments are connected to one another through the experiment nodes. The former can be created
standalone without an EVC tree. When connected to an `ExperimentNode`, the experiments gain access
to trials of other experiments by using method `ExperimentNode.fetch_trials`.

Helper functions are provided to fetch trials keeping the tree structure. Those can be helpful when
analyzing an EVC tree.

"""
import functools
import logging

from orion.core.evc.tree import TreeNode
from orion.storage.base import get_storage

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
        `Experiment` in read only mode.

    .. seealso::

        :py:class:`orion.core.evc.tree.TreeNode` for tree-specific attributes and methods.

    """

    __slots__ = (
        "name",
        "version",
        "_no_parent_lookup",
        "_no_children_lookup",
    ) + TreeNode.__slots__

    def __init__(self, name, version, experiment=None, parent=None, children=tuple()):
        """Initialize experiment node with item, experiment, parent and children

        .. seealso::
            :class:`orion.core.evc.tree.TreeNode` for information about the attributes
        """
        super(ExperimentNode, self).__init__(experiment, parent, children)
        self.name = name
        self.version = version

        self._no_parent_lookup = True
        self._no_children_lookup = True

    @property
    def item(self):
        """Get the experiment associated to the node

        Note that accessing `item` may trigger the lazy initialization of the experiment if it was
        not done already.
        """
        if self._item is None:
            # TODO: Find another way around the circular import
            import orion.core.io.experiment_builder as experiment_builder

            self._item = experiment_builder.load(name=self.name, version=self.version)
            self._item._node = self

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
            query = {"_id": self.item.refers.get("parent_id")}
            selection = {"name": 1, "version": 1}
            experiments = get_storage().fetch_experiments(query, selection)

            if experiments:
                parent = experiments[0]
                exp_node = ExperimentNode(
                    name=parent["name"], version=parent.get("version", 1)
                )
                self.set_parent(exp_node)
        return self._parent

    @property
    def children(self):
        """Get children of the experiment, empty list if no children

        .. note::

            The instantiation of an EVC tree is lazy, which means accessing the children of a node
            may trigger a call to database to build those children live.

        """
        if self._no_children_lookup:
            self._children = []
            self._no_children_lookup = False
            query = {"refers.parent_id": self.item.id}
            selection = {"name": 1, "version": 1}
            experiments = get_storage().fetch_experiments(query, selection)
            for child in experiments:
                self.add_children(
                    ExperimentNode(name=child["name"], version=child.get("version", 1))
                )

        return self._children

    @property
    def adapter(self):
        """Get the adapter of the experiment with respect to its parent"""
        return self.item.refers["adapter"]

    @property
    def tree_name(self):
        """Return a formatted name of the Node for a tree pretty-print."""
        if self.item is not None:
            return self.name + "-v{}".format(self.item.version)

        return self.name

    def fetch_lost_trials(self):
        """See :meth:`orion.core.evc.experiment.ExperimentNode.recurvise_fetch`"""
        return self.recurvise_fetch("fetch_lost_trials")

    def fetch_trials(self):
        """See :meth:`orion.core.evc.experiment.ExperimentNode.recurvise_fetch`"""
        return self.recurvise_fetch("fetch_trials")

    def fetch_pending_trials(self):
        """See :meth:`orion.core.evc.experiment.ExperimentNode.recurvise_fetch`"""
        return self.recurvise_fetch("fetch_pending_trials")

    def fetch_noncompleted_trials(self):
        """See :meth:`orion.core.evc.experiment.ExperimentNode.recurvise_fetch`"""
        return self.recurvise_fetch("fetch_noncompleted_trials")

    def fetch_trials_by_status(self, status):
        """See :meth:`orion.core.evc.experiment.ExperimentNode.recurvise_fetch`"""
        return self.recurvise_fetch("fetch_trials_by_status", status=status)

    def recurvise_fetch(self, fun_name, *args, **kwargs):
        """Fetch trials recursively in the EVC tree using the fetch function `fun_name`.

        Parameters
        ----------
        fun_name: callable
            Function name to call to fetch trials. The function must be an attribute of
            :class:`orion.core.worker.experiment.Experiment`

        *args:
            Positional arguments to pass to `fun_name`.

        **kwargs
            Keyword arguments to pass to `fun_name`.

        """

        def retrieve_trials(node, parent_or_children):
            """Retrieve the trials of a node/experiment."""
            fun = getattr(node.item, fun_name)
            # with_evc_tree needs to be False here or we will have an infinite loop
            trials = fun(*args, with_evc_tree=False, **kwargs)
            return dict(trials=trials, experiment=node.item), parent_or_children

        # get the trials of the parents
        parent_trials = None
        if self.parent is not None:
            parent_trials = self.parent.map(retrieve_trials, self.parent.parent)

        # get the trials of the children
        children_trials = self.map(retrieve_trials, self.children)
        children_trials.set_parent(parent_trials)

        adapt_trials(children_trials)

        return sum([node.item["trials"] for node in children_trials.root], [])


def _adapt_parent_trials(node, parent_trials_node, ids):
    """Adapt trials from the parent recursively

    .. note::

        To call with node.map(fct, node.parent) to connect with parents

    """
    # Ids from children are passed to prioritized them if they are also present in parent nodes.
    node_ids = (
        set(
            trial.compute_trial_hash(trial, ignore_lie=True, ignore_experiment=True)
            for trial in node.item["trials"]
        )
        | ids
    )
    if parent_trials_node is not None:
        adapter = node.item["experiment"].refers["adapter"]
        for parent in parent_trials_node.root:
            parent.item["trials"] = adapter.forward(parent.item["trials"])

            # if trial is in current exp, filter out
            parent.item["trials"] = [
                trial
                for trial in parent.item["trials"]
                if trial.compute_trial_hash(
                    trial, ignore_lie=True, ignore_experiment=True
                )
                not in node_ids
            ]

    return node.item, parent_trials_node


def _adapt_children_trials(node, children_trials_nodes):
    """Adapt trials from the children recursively

    .. note::

        To call with node.map(fct, node.children) to connect with children

    """
    ids = set(
        trial.compute_trial_hash(trial, ignore_lie=True, ignore_experiment=True)
        for trial in node.item["trials"]
    )

    for child in children_trials_nodes:
        adapter = child.item["experiment"].refers["adapter"]
        for subchild in child:  # Includes child itself
            subchild.item["trials"] = adapter.backward(subchild.item["trials"])

            # if trial is in current node, filter out
            subchild.item["trials"] = [
                trial
                for trial in subchild.item["trials"]
                if trial.compute_trial_hash(
                    trial, ignore_lie=True, ignore_experiment=True
                )
                not in ids
            ]

    return node.item, children_trials_nodes


def adapt_trials(trials_tree):
    """Adapt trials recursively so that they are all compatible with current experiment."""
    trials_tree.map(_adapt_children_trials, trials_tree.children)
    ids = set()
    for child in trials_tree.children:
        for trial in child.item["trials"]:
            ids.add(
                trial.compute_trial_hash(trial, ignore_lie=True, ignore_experiment=True)
            )
    trials_tree.map(
        functools.partial(_adapt_parent_trials, ids=ids), trials_tree.parent
    )
