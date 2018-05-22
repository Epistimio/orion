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
        """
        Initialize the ExperimentBranchBuilder by populating a list of the conflicts inside
        the two configurations.
        """
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
        # Loop through the conflicting space and identify problematic dimensions
        for dim in self.conflicting_space.values():
            # If the name is inside the space but not the value the dimensions has changed
            if dim.name in self.experiment_space:
                if dim not in self.experiment_space.values():
                    self.conflicts['changed'].append(dim)
            # If the name does not exist, it is a new dimension
            else:
                self.conflicts['new'].append(dim)

        # In the same vein, if any dimension of the current space is not inside
        # the conflicting space, it is missing
        for dim in self.experiment_space.values():
            if dim.name not in self.conflicting_space:
                self.conflicts['missing'].append(dim)

    def _create_dimensions_name_list(self):
        # Keep a list of the name of all the dimensions for each status
        self.new_dimensions_names = self._list_dimensions_names('new')
        self.missing_dimensions_names = self._list_dimensions_names('missing')
        self.changed_dimensions_names = self._list_dimensions_names('changed')

    def _list_dimensions_names(self, key):
        return list(map(lambda d: d.name, self.conflicts[key]))

    # API section

    def change_experiment_name(self, arg):
        """Make sure arg is a valid, non-conflicting name, and change the experiment's name to it"""
        if arg != self.conflicts['experiment']:
            self.conflicts['experiment'] = ''
            self.solved_conflicts['experiment'] = arg

    def add_dimension(self, name):
        """Add `name` dimension to the solved conflicts list"""
        prefixed_name = '/' + name
        if prefixed_name in self.new_dimensions_names:
            self._mark_as_solved(name, self.new_dimensions_names, 'new')
        elif prefixed_name in self.changed_dimensions_names:
            self._mark_as_solved(name, self.changed_dimensions_names, 'changed')

    def remove_dimension(self, name):
        """Remove `name` from the configuration and marks conflict as solved"""
        self._mark_as_solved(name, self.missing_dimensions_names, 'missing')

    def get_dimension_status(self, name):
        """Return the status of a dimension (unsolved/solved)"""
        is_in, key, value = self._is_dimension_in_dict('/' + name, self.conflicts)

        if is_in is False:
            return self._is_dimension_in_dict('/' + name, self.solved_conflicts)

        is_in = False
        return is_in, key, value

    def get_old_dimension_value(self, name):
        """Return the dimension from the parent experiment space"""
        if name in self.experiment_space:
            return self.experiment_space[name]

        return None

    def rename_dimension(self, args):
        """Change the name of old dimension to new dimension"""
        old, new = args
        if self._is_dimension_in_list(old, self.conflicts['missing']) and \
           self._is_dimension_in_list(new, self.conflicts['new']):
            self._mark_as_solved(old, self.missing_dimensions_names, 'missing')
            self._mark_as_solved(new, self.new_dimensions_names, 'new')

    def reset_dimension(self, arg):
        """Remove dimension `arg` from the solved conflicts"""
        is_in, status, value = self._is_dimension_in_dict('/' + arg, self.solved_conflicts)

        if is_in:
            self.solved_conflicts[status].remove(value)
            self.conflicts[status].append(value)
            self._create_dimensions_name_list()

    # Helper functions

    def _mark_as_solved(self, name, name_list, status):
        index = name_list.index('/' + name)
        name_list.pop(index)
        self._change_conflict_status(self.conflicts, self.solved_conflicts,
                                     status, index)

    def _change_conflict_status(self, source, destination, status, index):
        dim = source[status].pop(index)
        destination[status].append(dim)

    def _is_dimension_in_dict(self, name, conflicts_dict):
        for k in conflicts_dict:
            if k == 'experiment':
                continue

            is_in, value = self._is_dimension_in_list(name, conflicts_dict[k])
            if is_in:
                return True, k, value

        return False, None, None

    def _is_dimension_in_list(self, name, dimensions_list):
        for value in dimensions_list:
            if value.name == name:
                return True, value

        return False, None

    def _craft_bridge_api_package(self):
        pass
