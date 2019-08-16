#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.experiment_branch_builder.py` -- Module offering an API to solve conflicts
==============================================================================================

.. module:: experiment_branch_builder
   :platform: Unix
   :synopsis: Create a list of adapters from the conflicts between an experiment and its parent.

Conflicts between two experiments arise when those have different configuration but have the same
name. Solving these conflicts require the creation of adapters to bridge from the parent experiment
and the child experiment.

 .. seealso::

    :mod:`orion.core.evc.conflicts`
    :mod:`orion.core.evc.adapters`

"""

import logging

from orion.algo.space import Dimension
from orion.core.evc import conflicts
from orion.core.evc.adapters import CompositeAdapter


log = logging.getLogger(__name__)


# pylint: disable=too-many-public-methods
class ExperimentBranchBuilder:
    """Build a new configuration for the experiment based on parent config."""

    def __init__(self, conflicts, branching_configuration):
        """
        Initialize the ExperimentBranchBuilder by populating a list of the conflicts inside
        the two configurations.
        """
        self.auto_resolution = branching_configuration.pop('auto_resolution', None)

        if self.auto_resolution is not None:
            log.info("Auto-resolution is deprecated and will be removed in v0.2.0.")
            self.auto_resolution = None

        self.manual_resolution = branching_configuration.pop('manual_resolution', False)
        self.conflicts = conflicts
        self.conflicting_config.update(branching_configuration)
        self.resolve_conflicts()

    @property
    def experiment_config(self):
        """Get configuration of the parent experiment"""
        return self.conflicts.get()[0].old_config

    @property
    def conflicting_config(self):
        """Get configuration of the child experiment"""
        return self.conflicts.get()[0].new_config

    def resolve_conflicts(self, silence_errors=True):
        """Automatically resolve conflicts unless manual-resolution is True."""
        ith_conflict = 0

        while ith_conflict < len(self.conflicts.get()):

            # TODO: Meh, ugly
            conflict = self.conflicts.conflicts[ith_conflict]

            resolution = self.conflicts.try_resolve(
                conflict, silence_errors=silence_errors,
                **conflict.get_marked_arguments(self.conflicts))

            if resolution and (self.manual_resolution and not resolution.is_marked):
                self.conflicts.revert(resolution)

            ith_conflict += 1

    # API section
    @property
    def is_resolved(self):
        """Return True if all the current conflicts have been resolved"""
        return self.conflicts.are_resolved

    def change_experiment_name(self, name):
        """Change the child's experiment name to `name`

        Parameters
        ----------
        name: str
           New name for the child experiment. Must be different from the parent's name

        Raises
        ------
        ValueError
            If name already exists in database for current user.
        RuntimeError
            If there is no code change conflict left to resolve.

        """
        exp_name_conflicts = self.conflicts.get_remaining([conflicts.ExperimentNameConflict])
        if not exp_name_conflicts:
            raise RuntimeError('No experiment name conflict to solve')

        self.conflicts.try_resolve(exp_name_conflicts[0], name)

    def set_code_change_type(self, change_type):
        """Set code change type

        Parameters
        ----------
        change_type: string
            One of the types defined in `orion.core.evc.adapters.CodeChange.types`.

        Raises
        ------
        ValueError
            If change_type is not in `orion.core.evc.adapters.CodeChange.types`.
        RuntimeError
            If there is no code change conflict left to resolve.

        """
        code_conflicts = self.conflicts.get_remaining([conflicts.CodeConflict])
        if not code_conflicts:
            raise RuntimeError('No code conflicts to solve')

        self.conflicts.try_resolve(code_conflicts[0], change_type=change_type)

    def set_cli_change_type(self, change_type):
        """Set cli change type

        Parameters
        ----------
        change_type: string
            One of the types defined in `orion.core.evc.adapters.CommandLineChange.types`.

        Raises
        ------
        ValueError
            If change_type is not in `orion.core.evc.adapters.CommandLineChange.types`.
        RuntimeError
            If there is no cli conflict left to resolve.

        """
        cli_conflicts = self.conflicts.get_remaining([conflicts.CommandLineConflict])
        if not cli_conflicts:
            raise RuntimeError('No command line conflicts to solve')

        self.conflicts.try_resolve(cli_conflicts[0], change_type)

    def set_script_config_change_type(self, change_type):
        """Set script config change type

        Parameters
        ----------
        change_type: string
            One of the types defined in `orion.core.evc.adapters.ScriptConfigChange.types`.

        Raises
        ------
        ValueError
            If change_type is not in `orion.core.evc.adapters.ScriptConfigChange.types`.
        RuntimeError
            If there is no script config conflict left to resolve.

        """
        script_config_conflicts = self.conflicts.get_remaining([conflicts.ScriptConfigConflict])
        if not script_config_conflicts:
            raise RuntimeError('No script\'s config conflicts to solve')

        self.conflicts.try_resolve(script_config_conflicts[0], change_type)

    def set_algo(self):
        """Set algorithm resolution

        Raises
        ------
        RuntimeError
            If there is no algorithm conflict left to resolve.

        """
        algo_conflicts = self.conflicts.get_remaining([conflicts.AlgorithmConflict])
        if not algo_conflicts:
            raise RuntimeError('No algo conflict to solve')

        self.conflicts.try_resolve(algo_conflicts[0])

    def add_dimension(self, name, default_value=Dimension.NO_DEFAULT_VALUE):
        """Add dimension with given `name`

        Only dimensions with conflict type `NewDimensionConflict` or `ChangedDimensionConflict` may
        be added.

        Parameters
        ----------
        name: str
            Name of the dimension to add
        default_value: object
            Default value for the new dimension. Defaults to `Dimension.NO_DEFAULT_VALUE`.
            If conflict is ChangedDimensionConflict, default_value is ignored.

        Raises
        ------
        ValueError
            If name is not present in non-resolved conflicts or if default_value is invalid for the
            corresponding dimension.

        """
        conflict = self.conflicts.get_remaining(
            [conflicts.NewDimensionConflict, conflicts.ChangedDimensionConflict],
            dimension_name=name)[0]

        if isinstance(conflict, conflicts.NewDimensionConflict):
            self.conflicts.try_resolve(conflict, default_value=default_value)
        else:
            self.conflicts.try_resolve(conflict)

    def remove_dimension(self, name, default_value=Dimension.NO_DEFAULT_VALUE):
        """Remove dimension with given `name`

        Only dimensions with conflict type `MissingDimensionConflict` may be added.

        Parameters
        ----------
        name: str
            Name of the dimension to add
        default_value: object
            Default value for the missing dimension. Defaults to `Dimension.NO_DEFAULT_VALUE`.

        Raises
        ------
        ValueError
            If name is not present in non-resolved conflicts or if default_value is invalid for the
            corresponding dimension.

        """
        conflict = self.conflicts.get_remaining(
            [conflicts.MissingDimensionConflict], dimension_name=name)[0]

        self.conflicts.try_resolve(conflict, default_value=default_value)

    def rename_dimension(self, old_name, new_name):
        """Rename dimension `old_name` to `new_name`

        Only dimensions with conflict type `MissingDimensionConflict` may be renamed,
        and it can only be renamed to dimensions with conflict type `NewDimensionConflict`.

        Parameters
        ----------
        old_name: str
            Name of the dimension to rename
        new_name: str
            Name of the target dimension

        Raises
        ------
        ValueError
            If name is not present in non-resolved conflicts.

        Notes
        -----
        This may create a new conflict if the target dimension has a different prior.

        """
        potential_conflicts = self.conflicts.get_remaining(
            [conflicts.MissingDimensionConflict], dimension_name=old_name)

        assert len(potential_conflicts) == 1, ("Many missing dimensions with the same name: "
                                               "{}".format(", ".join(potential_conflicts)))

        old_dim_conflict = potential_conflicts[0]

        potential_conflicts = self.conflicts.get_remaining(
            [conflicts.NewDimensionConflict], dimension_name=new_name)

        assert len(potential_conflicts) == 1, ("Many new dimensions with the same name: "
                                               "{}".format(", ".join(potential_conflicts)))

        new_dim_conflict = potential_conflicts[0]

        self.conflicts.try_resolve(old_dim_conflict, new_dimension_conflict=new_dim_conflict)

    def reset(self, name):
        """Revert a resolution and reset its corresponding conflicts

        Parameters
        ----------
        name: str
            String representing the resolution as provided in the prompt

        Raises
        ------
        ValueError
            If name does not correspond to one of the current resolutions.

        Notes
        -----
        Side-effect conflicts generated by a reverted resolution will be deleted.

        """
        self.conflicts.revert(name)

    def create_adapters(self):
        """Return a list of adapters for every resolution"""
        adapters = []
        for resolution in self.conflicts.get_resolutions():
            adapters += resolution.get_adapters()

        return CompositeAdapter(*adapters)
