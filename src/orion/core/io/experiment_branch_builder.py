#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.experiment_branch_builder.py` -- Module building the
difference between a parent experience and its branching child.
========================================================================
.. module:: experiment_branch_builder
   :platform: Unix
   :synopsis: Gets a conflicting config regarding a given experiment and
   handles the solving of the different conflicts

"""

import logging

from orion.core.io.space_builder import SpaceBuilder

log = logging.getLogger(__name__)


class ExperimentBranchBuilder:
    """Build a new configuration for the experiment based on parent config."""


    def __init__(self, experiment_config, conflicting_config):
        self.experiment_config = experiment_config
        self.conflicting_config = conflicting_config

        self.conflicts = {'changed': [], 'new': [], 'missing': [], 'experiment': ''}
        self.solved_conflicts = {'changed': [], 'new': [], 'missing': [], 'experiment': ''}

        self._build_spaces()
        self._find_conflicts()
        self._create_dimensions_name_list()
        
        self.conflicts['experiment'] = self.experiment_config['name']

    def _build_spaces(self):
        user_args = self.conflicting_config['metadata']['user_args']
        experiment_args = self.experiment_config['metadata']['user_args']

        self.experiment_space = SpaceBuilder().build_from(experiment_args)
        self.conflicting_space = SpaceBuilder().build_from(user_args)

    def _find_conflicts(self):
        for d in self.conflicting_space.values():
            if d.name in self.experiment_space:
                if d not in self.experiment_space.values():
                    self.conflicts['changed'].append(d)
            else:
                self.conflicts['new'].append(d)

        for d in self.experiment_space.values():
            if d.name not in self.conflicting_space:
                self.conflicts['missing'].append(d)

    def _create_dimensions_name_list(self):
        self.new_dimensions_names = self._list_dimensions_names('new')
        self.missing_dimensions_names = self._list_dimensions_names('missing')
        self.changed_dimensions_names = self._list_dimensions_names('changed')

    def _list_dimensions_names(self, key):
        return list(map(lambda d: d.name, self.conflicts[key]))

    def change_experiment_name(self, arg):
        """Make sure `arg` is a valid, non-conflicting name, and change the experiment's name to it"""
        if arg != self.conflicts['experiment']:
            self.conflicts['experiment'] = ''
            self.solved_conflicts['experiment'] = arg

    def add_dimension(self, name):
        """Add `name` dimension to the solved conflicts list"""
        index = self.new_dimensions_names.index('/' + name)
        self.new_dimensions_names.pop(index)
        dim = self.conflicts['new'].pop(index)
        self.solved_conflicts['new'].append(dim)

    def get_dimension_status(self, name):
        """Return the status of a dimension (unsolved/solved)"""
        is_in, key, value = self._is_dimension_in_dict(name, self.conflicts)
        
        if is_in == False:
            return self._is_dimension_in_dict(name, self.solved_conflicts)

        is_in = False
        return is_in, key, value

    def _is_dimension_in_dict(self, name, conflicts_dict):
        for k in conflicts_dict:
            for value in conflicts_dict[k]:
                return True, k, value
        
        return False, None, None

    def get_old_dimension_value(self, name):
        if name in self.experiment_space:
            return self.experiment_space[name]

    def _craft_bridge_api_package(self):
        pass
