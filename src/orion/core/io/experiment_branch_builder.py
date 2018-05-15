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


class Conflict:
    """Represent a single conflict inside the configuration"""

    def __init__(self, status, dimension):
        self.is_solved = False
        self.dimension = dimension
        self.status = status
        self.operation = None


class ExperimentBranchBuilder:
    """Build a new configuration for the experiment based on parent config."""

    def __init__(self, experiment_config, conflicting_config):
        """
        Initialize the ExperimentBranchBuilder by populating a list of the conflicts inside
        the two configurations.
        """
        self.experiment_config = experiment_config
        self.conflicting_config = conflicting_config

        self.conflicts = []

        self._build_spaces()
        self._find_conflicts()

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
                    self.conflicts.append(Conflict('changed', dim))
            # If the name does not exist, it is a new dimension
            else:
                self.conflicts.append('new', dim)

        # In the same vein, if any dimension of the current space is not inside
        # the conflicting space, it is missing
        for dim in self.experiment_space.values():
            if dim.name not in self.conflicting_space:
                self.conflicts.append('missing', dim)

    # API section

    def change_experiment_name(self, arg):
        """Make sure arg is a valid, non-conflicting name, and change the experiment's name to it"""
        if arg != self.experiment_config['name']:
            self.conflicting_config['name'] = arg

    def add_dimension(self, name):
        """Add `name` dimension to the solved conflicts list"""
        self._mark_as_solved(name, ['new', 'changed'])

    def remove_dimension(self, name):
        """Remove `name` from the configuration and marks conflict as solved"""
        self._mark_as_solved(name, ['missing'])

    def get_dimension_conflict(self, name):
        prefixed_name = '/' + name
        index = list(map(lambda c: c.dimension.name, self.conflicts)).index(prefixed_name)
        return self.conflicts[index]

    def get_old_dimension_value(self, name):
        """Return the dimension from the parent experiment space"""
        if name in self.experiment_space:
            return self.experiment_space[name]

        return None

    def rename_dimension(self, args):
        """Change the name of old dimension to new dimension"""
        old, new = args
        self._mark_as_solved('/' + old, ['missing'])
        self._mark_as_solved('/' + new, ['new'])

    def reset_dimension(self, arg):
        self._mark_as(arg, ['missing', 'new', 'changed'], False)

    def get_conflicts_with_solved_status(self, wants_solved=False):
        return filter(lambda c: c.is_solved is wants_solved, self.conflicts)

    def get_filtered_conflicts(self, status):
        return filter(lambda c: c.status in status, self.conflicts)

    def filter_conflicts(self, filter_function):
        return filter(filter_function, self.conflicts)
    # Helper functions

    def _mark_as_solved(self, name, status):
        self._mark_as(name, status, True)

    def _mark_as(self, name, status, is_solved):
        prefixed_name = '/' + name

        conflicts = list(self.get_filtered_conflicts(status))
        index = list(map(lambda c: c.dimension.name, conflicts)).index(prefixed_name)
        conflicts[index].is_solved = is_solved

    def _craft_bridge_api_package(self):
        pass
