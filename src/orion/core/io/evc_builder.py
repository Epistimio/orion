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
from orion.core.evc.experiment import ExperimentNode
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
    def connect_to_version_control_tree(self, experiment):
        """Build the EVC and connect the experiment to it"""
        experiment_node = ExperimentNode(experiment.name, experiment=experiment)
        experiment.connect_to_version_control_tree(experiment_node)

    def build_view_from(self, cmdargs):
        """Build an experiment view based on global config and connect it to the EVC"""
        experiment_view = ExperimentBuilder().build_view_from(cmdargs)
        self.connect_to_version_control_tree(experiment_view)

        return experiment_view

    def build_from(self, cmdargs):
        """Build an experiment based on config and connect it to the EVC"""
        experiment = ExperimentBuilder().build_from(cmdargs)
        self.connect_to_version_control_tree(experiment)

        return experiment

    def build_from_config(self, config):
        """Build an experiment based on given config and connect it to the EVC"""
        experiment = ExperimentBuilder().build_from_config(config)
        self.connect_to_version_control_tree(experiment)

        return experiment
