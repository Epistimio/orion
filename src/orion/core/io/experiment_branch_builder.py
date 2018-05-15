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
        self.operations = {}
        self._operations_mapping = {'add': self._add_adaptor,
                                    'rename': self._rename_adaptor,
                                    'remove': self._remove_adaptor}

        self._build_spaces()
        self._find_conflicts()

        branching_name = conflicting_config.pop('branch', None)
        if branching_name is not None:
            self.change_experiment_name(branching_name)

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
                self.conflicts.append(Conflict('new', dim))

        # In the same vein, if any dimension of the current space is not inside
        # the conflicting space, it is missing
        for dim in self.experiment_space.values():
            if dim.name not in self.conflicting_space:
                self.conflicts.append(Conflict('missing', dim))

    # API section

    def change_experiment_name(self, arg):
        """Make sure arg is a valid, non-conflicting name, and change the experiment's name to it"""
        if arg != self.experiment_config['name']:
            self.conflicting_config['name'] = arg

    def add_dimension(self, name):
        """Add `name` dimension to the solved conflicts list"""
        conflict = self._mark_as_solved(name, ['new', 'changed'])
        self._put_operation('add', (conflict))

    def remove_dimension(self, name):
        """Remove `name` from the configuration and marks conflict as solved"""
        conflict = self._mark_as_solved(name, ['missing'])
        self._put_operation('remove', (conflict))

    def rename_dimension(self, args):
        """Change the name of old dimension to new dimension"""
        old, new = args

        old_index, missing_conflicts = self._assert_has_status(old, 'missing')
        new_index, new_conflicts = self._assert_has_status(new, 'new')

        old_conflict = missing_conflicts[old_index]
        new_conflict = new_conflicts[new_index]

        old_conflict.is_solved = True
        new_conflict.is_solved = True
        self._put_operation('rename', (old_conflict, new_conflict))

    def reset_dimension(self, arg):
        conflict = self._mark_as(arg, ['missing', 'new', 'changed'], False)
        self._remove_from_operations(conflict)

    def get_dimension_conflict(self, name):
        prefixed_name = '/' + name
        index = list(map(lambda c: c.dimension.name, self.conflicts)).index(prefixed_name)
        return self.conflicts[index]

    def get_old_dimension_value(self, name):
        """Return the dimension from the parent experiment space"""
        if name in self.experiment_space:
            return self.experiment_space[name]

        return None

    def filter_conflicts_with_solved_state(self, wants_solved=False):
        return self.filter_conflicts(lambda c: c.is_solved is wants_solved)

    def filter_conflicts_with_status(self, status):
        return self.filter_conflicts(lambda c: c.status in status)

    def filter_conflicts(self, filter_function):
        return filter(filter_function, self.conflicts)

    def create_adaptors(self):
        adaptors = []
        for operation in self.operations:
            for conflict in self.operations[operation]:
                adaptors.append(self._operations_mapping[operation](conflict))

    # Helper functions

    def _mark_as_solved(self, name, status):
        return self._mark_as(name, status, True)

    def _mark_as(self, name, status, is_solved):
        index, conflicts = self._assert_has_status(name, status)
        conflict = conflicts[index]
        conflict.is_solved = is_solved

        return conflict

    def _assert_has_status(self, name, status):
        prefixed_name = '/' + name
        conflicts = list(self.filter_conflicts_with_status(status))
        index = list(map(lambda c: c.dimension.name, conflicts)).index(prefixed_name)

        return index, conflicts

    def _put_operation(self, operation_name, args):
        if operation_name not in self.operations:
            self.operations[operation_name] = []

        if args not in self.operations[operation_name]:
            self.operations[operation_name].append(args)

    def _remove_from_operations(self, arg):
        for operation in self.operations:
            if operation == 'rename':
                for value in self.operations[operation]:
                    old, new = value
                    if arg in value:
                        old.is_solved = False
                        new.is_solved = False
                        self.operations[operation].remove(value)

            elif arg in self.operations[operation]:
                arg.is_solved = False
                self.operations[operation].remove(arg)

    # TODO Create Adaptor instances
    def _add_adaptor(self, conflict):
        pass

    def _rename_adaptor(self, conflict):
        pass

    def _remove_adaptor(self, conflict):
        pass
