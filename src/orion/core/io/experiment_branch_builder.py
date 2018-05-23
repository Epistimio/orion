#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.experiment_branch_builder.py` -- Module offering an API to solve conflicts
==============================================================================================

.. module:: experiment_branch_builder
   :platform: Unix
   :synopsis: Create a list of Adapters from the conflicts between an experiment and it's parent.

Conflicts between two experiments arise when these experiments have different
spaces but have the same name. Solving these conflicts require the creation of
Adapters to bridge from the parent experiment and the child experiment.

Conflicting dimensions can be in one of three different states :
    * New : this dimension is not present in the parent's space
    * Changed : this dimension's prior has changed between the parent's and the child's space
    * Missing : this dimension is not present in the child's space

To solve these conflicts, the builder object offers a public API to add, remove or rename
dimensions.

For more info on Adapters :
    ..seealso::
        :meth:`orion.core.evc.Adapters.BaseAdapter`
"""

import logging
import re

import orion.core.evc.adapters as Adapters
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)


def _create_param(dimension):
    return Trial.Param(name=dimension.name, _type=dimension.type, value=dimension.default_value)


def _get_expression(dim, args):
    return list(filter(lambda arg: arg[1:].startswith(dim.name[1:]), args))[0].split('~')[1]


# pylint: disable=too-few-public-methods
class Conflict:
    """Represent a single conflict inside the configuration"""

    def __init__(self, status, dimension, expression):
        """Init conflict"""
        self.is_solved = False
        self.dimension = dimension
        self.status = status
        self.expression = expression


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
        self._operations_mapping = {'add': self._add_adapter,
                                    'rename': self._rename_adapter,
                                    'remove': self._remove_adapter}

        self.special_keywords = {'~new': 'new',
                                 '~changed': 'changed',
                                 '~missing': 'missing'}

        self.commandline_keywords = {'~+': [], '~-': [], '~>': []}

        self.cl_keywords_functions = {'~+': self.add_dimension,
                                      '~-': self.remove_dimension,
                                      '~>': self.rename_dimension}

        self.cl_keywords_re = {'~+': re.compile(r'([a-zA-Z_]+)~+'),
                               '~-': re.compile(r'([a-zA-Z_]+)~-'),
                               '~>': re.compile(r'([a-zA-Z_]+)~>([a-zA-Z_]+)')}

        self.user_args = self.conflicting_config['metadata']['user_args']
        self.experiment_args = self.experiment_config['metadata']['user_args']

        self._interpret_commandline()
        self._build_spaces()
        self._find_conflicts()
        self._solve_commandline_conflicts()

        branching_name = conflicting_config.pop('branch', None)
        if branching_name is not None:
            self.change_experiment_name(branching_name)

    def _interpret_commandline(self):
        to_delete = []
        for arg in self.user_args:
            for keyword in self.commandline_keywords:
                if keyword in arg:
                    self.commandline_keywords[keyword].append(arg)

                    index = self.user_args.index(arg)
                    if keyword == '~+':
                        self.user_args[index] = arg.replace(keyword, '~')
                    else:
                        to_delete.append(index)

        for i in sorted(to_delete, reverse=True):
            del self.user_args[i]

    def _build_spaces(self):
        self.experiment_space = SpaceBuilder().build_from(self.experiment_args)
        self.conflicting_space = SpaceBuilder().build_from(self.user_args)

    def _find_conflicts(self):
        # Loop through the conflicting space and identify problematic dimensions
        for dim in self.conflicting_space.values():
            # If the name is inside the space but not the value the dimensions has changed
            if dim.name in self.experiment_space:
                if dim not in self.experiment_space.values():
                    self.conflicts.append(Conflict('changed', dim, _get_expression(dim,
                                                                                   self.user_args)))
            # If the name does not exist, it is a new dimension
            else:
                self.conflicts.append(Conflict('new', dim, _get_expression(dim, self.user_args)))

        # In the same vein, if any dimension of the current space is not inside
        # the conflicting space, it is missing
        for dim in self.experiment_space.values():
            if dim.name not in self.conflicting_space:
                self.conflicts.append(Conflict('missing', dim,
                                               _get_expression(dim, self.experiment_args)))

    def _solve_commandline_conflicts(self):
        for keyword in self.commandline_keywords:
            for dimension in self.commandline_keywords[keyword]:
                value = self.cl_keywords_re[keyword].findall(dimension)
                self.cl_keywords_functions[keyword](*value)

    # API section

    @property
    def is_solved(self):
        """Return True if all the current conflicts have been solved"""
        solved = list(self.filter_conflicts(lambda c: c.is_solved))

        experiment_name = self.experiment_config['name']
        child_name = self.conflicting_config['name']
        return len(solved) == len(self.conflicts) and experiment_name != child_name

    def change_experiment_name(self, name):
        """Change the child's experiment name to `name`

        Parameters
        ----------
        name: str
           New name for the child experiment. Must be different from the parent's name

        """
        if name != self.experiment_config['name']:
            self.conflicting_config['name'] = name

    def add_dimension(self, args):
        """Add the dimensions whose name's are inside the args input string to the
        child's space by solving their respective conflicts.

        Only dimensions with the `new` or `changed` conflicting state may be added.

        Parameters
        ----------
        args: str
            String containing dimensions' name separated by a whitespace.

        """
        self._do_basic(args, ['new', 'changed'], 'add')

    def remove_dimension(self, name):
        """Remove the dimensions whose name's are inside the args input string from the
        child's space by solving their respective conflicts.

        Only dimensions with the `missing` conflicting state may be removed.

        Parameters
        ----------
        args: str
            String containing dimensions' name separated by a whitespace.

        """
        self._do_basic(name, ['missing'], 'remove')

    def _do_basic(self, name, status, operation):
        for _name in self._get_names(name, status):
            conflict = self._mark_as_solved(_name, status)
            self._put_operation(operation, (conflict))

    def reset_dimension(self, arg):
        """Remove the dimensions whose name's are inside the args input string from the
        solved conflicts list.

        Parameters
        ----------
        args: str
            String containing dimensions' name separated by a whitespace.

        """
        status = ['missing', 'new', 'changed']
        for _name in self._get_names(arg, status):
            conflict = self._mark_as(arg, status, False)
            self._remove_from_operations(conflict)

    def rename_dimension(self, args):
        """Rename the first dimension inside the args tuple from the parent's space
        to the second dimension inside the args tuple from the child's space.

        Only a `missing` dimension can be renamed. It can only be renamed to a `new`
        dimension.

        Parameters
        ----------
        args: tuple of two str
            Tuple containing the old dimension's name and the new dimension's name (in this order)

        """
        old, new = args

        old_index, missing_conflicts = self._assert_has_status(old, 'missing')
        new_index, new_conflicts = self._assert_has_status(new, 'new')

        old_conflict = missing_conflicts[old_index]
        new_conflict = new_conflicts[new_index]

        if new_conflict.is_solved or old_conflict.is_solved:
            raise ValueError('Only dimensions with an unsolved conflict may be renamed')

        old_conflict.is_solved = True
        new_conflict.is_solved = True
        self._put_operation('rename', (old_conflict, new_conflict))

    def get_dimension_conflict(self, name):
        """Return the conflict object related to the dimension of name `name`

        Parameters
        ----------
        name: str
            Name of the dimension

        """
        prefixed_name = '/' + name
        index = list(map(lambda c: c.dimension.name, self.conflicts)).index(prefixed_name)
        return self.conflicts[index]

    def get_old_dimension_value(self, name):
        """Return the Dimension object with name `name` from the parent experiment

        Parameters
        ----------
        name: str
            Name of the dimension

        """
        if name in self.experiment_space:
            return self.experiment_space[name]

        return None

    def filter_conflicts(self, filter_function):
        """Return a sublist of the conflicts filtered by the filter_function

        Parameters
        ----------
        filter_function: callable
            Function which returns a boolean for the `filter` call

        """
        return filter(filter_function, self.conflicts)

    def create_adapters(self):
        """Return a list of Adapters for every single dimension"""
        adapters = []
        for operation in self.operations:
            for conflict in self.operations[operation]:
                adapters.append(self._operations_mapping[operation](conflict))

        return Adapters.CompositeAdapter(*adapters)

    # Helper functions
    def _get_names(self, name, status):
        args = name.split(' ')
        names = []

        for arg in args:
            if arg in self.special_keywords:
                self._extend_special_keywords(arg, names)
            elif '*' in arg:
                self._extend_wildcard(arg, names, status)
            else:
                names = [arg]

        return names

    def _extend_special_keywords(self, arg, names):
        conflicts = self._filter_conflicts_status([self.special_keywords[arg]])
        names.extend(list(map(lambda c: c.dimension.name[1:], conflicts)))

    def _extend_wildcard(self, arg, names, status):
        prefix = '/' + arg.split('*')[0]
        filtered_conflicts = self.filter_conflicts(lambda c:
                                                   c.dimension.name
                                                   .startswith(prefix) and c.status in status)
        names.extend(list(map(lambda c: c.dimension.name[1:], filtered_conflicts)))

    def _mark_as_solved(self, name, status):
        return self._mark_as(name, status, True)

    def _mark_as(self, name, status, is_solved):
        index, conflicts = self._assert_has_status(name, status)
        conflict = conflicts[index]
        conflict.is_solved = is_solved

        return conflict

    def _assert_has_status(self, name, status):
        prefixed_name = '/' + name
        conflicts = list(self._filter_conflicts_status(status))
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

    def _filter_conflicts_status(self, status):
        def filter_status(conflict):
            """Filter the conflict having status inside the `status` list"""
            return conflict.status in status

        return self.filter_conflicts(filter_status)

    def _add_adapter(self, conflict):
        if conflict.status == 'changed':
            adapter = self._changed_adapter(conflict)
        else:
            adapter = Adapters.DimensionAddition(_create_param(conflict.dimension))

        return adapter

    def _changed_adapter(self, dimensions):
        if isinstance(dimensions, tuple):
            old_prior, new_prior = [c.expression for c in dimensions]
            return Adapters.DimensionPriorChange(dimensions[1].dimension.name, old_prior, new_prior)

        dim = self.get_old_dimension_value(dimensions.dimension.name)
        old_conflict = Conflict("", dim, _get_expression(dim, self.experiment_args))
        return self._changed_adapter((old_conflict, dimensions))

    def _rename_adapter(self, conflict):
        old, new = conflict
        old_dim = old.dimension
        new_dim = new.dimension

        rename = Adapters.DimensionRenaming(old_dim.name, new_dim.name)

        if old.expression != new.expression:
            rename = Adapters.CompositeAdapter(rename, self._changed_adapter(conflict))

        return rename

    @staticmethod
    def _remove_adapter(conflict):
        return Adapters.DimensionDeletion(_create_param(conflict.dimension))
