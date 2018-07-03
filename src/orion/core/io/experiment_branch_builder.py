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

    def __init__(self, conflicts, auto_resolution=False):
        """
        Initialize the ExperimentBranchBuilder by populating a list of the conflicts inside
        the two configurations.
        """
        self.auto_resolution = auto_resolution
        self.conflicts = conflicts
        self.resolve_conflicts()

    @property
    def experiment_config(self):
        """Get configuration of the parent experiment"""
        return self.conflicts.get()[0].old_config

    @property
    def conflicting_config(self):
        """Get configuration of the child experiment"""
        return self.conflicts.get()[0].new_config

    # def _init_mappings(self):

    #     self.cl_keywords_re = {'~+': re.compile(r'(.*)~\+'),
    #                            '~-': re.compile(r'(.*)~\-'),
    #                            '~>': re.compile(r'(.*)~\>([a-zA-Z_]+)')}

    # def _build_spaces(self):
    #     self.experiment_space = SpaceBuilder().build_from(self.experiment_args)
    #     self.extended_experiment_args.extend(_concat_key_value(SpaceBuilder().config_expressions))

    #     self.conflicting_space = SpaceBuilder().build_from(self._curate_user_args(self.user_args))

    #     # self.conflicting_space = SpaceBuilder().build_from(self.user_args)
    #     self.extended_user_args.extend(_concat_key_value(SpaceBuilder().config_expressions))

    # def _curate_user_args(self, user_args):
    #     curated_user_args = []
    #     for user_arg in user_args:
    #         if '~>' in user_arg:
    #             continue
    #             # value = self.cl_keywords_re['~>'].findall(user_arg)
    #             # if value:
    #             #     old_name, new_name = value[0]
    #             #     marker = "{}~>{}".format(old_name.lstrip("-"), new_name)
    #             #     user_arg = user_arg.replace(marker, new_name)

    #         # TODO: replace with proper regex
    #         elif "~+" in user_arg:
    #             user_arg = user_arg.replace("~+", "~")
    #         # TODO: replace with proper regex
    #         elif "~-" in user_arg:
    #             continue

    #         curated_user_args.append(user_arg)

    #     return curated_user_args

    # def detect_conflicts(self):
    #     self._build_spaces()

    #     self.conflicts.add(
    #         conflicts.ExperimentNameConflict(self.conflicting_config, self.experiment_config))

    #     self._detect_conflicts_in_dimensions()
    #     self._detect_missing_dimensions()
    #     self._detect_conflicts_in_algo()
    #     self._detect_conflicts_in_code()

    # def resolve_commandline_conflicts(self):

    #     # Use while-loop because some resolution might add new conflicts and a for-loop would not
    #     # catch this.
    #     ith_conflict = 0

    #     while ith_conflict < len(self.conflicts.get()):

    #         # TODO: Meh, ugly
    #         conflict = self.conflicts.conflicts[ith_conflict]

    #         resolution = self.try_resolve(
    #             conflict, silence_errors=False,
    #             **self._get_marked_arguments(self.conflicts.get()))

    #        if resolution and not (self.auto_resolution or self._resolution_is_marked(resolution)):
    #             self.resolutions.revert(resolution)

    #         ith_conflict += 1

    def resolve_conflicts(self, silence_errors=True):
        """Automatically resolve conflicts if auto-resolution is True or resolutions are marked."""
        ith_conflict = 0

        while ith_conflict < len(self.conflicts.get()):

            # TODO: Meh, ugly
            conflict = self.conflicts.conflicts[ith_conflict]

            resolution = self.conflicts.try_resolve(
                conflict, silence_errors=silence_errors,
                **conflict.get_marked_arguments(self.conflicts))

            if resolution and not (self.auto_resolution or resolution.is_marked):
                self.conflicts.revert(resolution)

            ith_conflict += 1

    # def _get_marked_arguments(self, conflict):
    #     for fct in [self._get_renamed_marked_arguments,
    #                 self._get_missing_marked_arguments]:
    #         marked_arguments = fct(conflict)
    #         if marked_arguments:
    #             return marked_arguments

    #     return marked_arguments

    # def _get_missing_marked_arguments(self, conflict):
    #     if not isinstance(conflict, conflicts.MissingDimensionConflict):
    #         return {}

    #     return {}

    # def _get_renamed_marked_arguments(self, conflict):
    #     if not isinstance(conflict, conflicts.MissingDimensionConflict):
    #         return {}

    #     marker = '{}{}'.format(conflict.dimension.name.lstrip("/"),
    #                            resolutions.RenameDimensionResolution.MARKER)
    #     user_args = [user_arg for user_arg in self.extended_user_args if marker in user_arg]
    #     if user_args:
    #         new_dimension_name = self.cl_keywords_re['~>'].findall(user_args[0])[0][1]

    #         try:
    #             conflict = self.conflicts.get([conflicts.NewDimensionConflict],
    #                                           dimension_name=new_dimension_name)[0]
    #         except ValueError as e:
    #             if "Dimension name '{}' not found".format(new_dimension_name) not in str(e):
    #                 return {}

    #         new_dimension = copy.deepcopy(conflict.dimension)
    #         new_dimension.name = "/" + new_dimension_name
    #         conflict = conflicts.MissingDimensionConflict(
    #             self.conflicting_config, new_dimension, conflict.prior)
    #         self.conflicts.add(conflict)

    #         print(new_dimension.get_string())
    #         self.extended_user_args.append("-" + new_dimension.get_string())

    #         # If dimension was already added
    #         if conflict.is_resolved:
    #             self.reset(str(conflict.resolution))

    #         return {'new_dimension_conflict': conflict}

    #     return {}

    # def _resolution_is_marked(self, resolution):
    #     if resolution.MARKER:
    #         # TODO: Detect marked renaming and resolve properly
    #         is_marked_by_user = any(arg.lstrip("-").startswith(resolution.prefix)
    #                                 for arg in self.extended_user_args)
    #     else:
    #         is_marked_by_user = (_convert_arg_to_namespace(resolution.ARGUMENT)
    #                              in self.conflicting_config)

    #     is_algo = isinstance(resolution, resolutions.AlgorithmResolution)

    #     return is_marked_by_user or is_algo

    # API section
    @property
    def is_resolved(self):
        """Return True if all the current conflicts have been resolved"""
        return self.conflicts.are_resolved

    # def set_resolution(self, resolution_string):
    #     if RenameDimensionResolution.MARKER in resolution_string:
    #         pass
    #     elif resolution_string.beginswith(AlgorithmResolution.ARGUMENT):
    #         pass
    #     elif resolution_string.beginswith(CodeChange.ARGUMENT):
    #         pass
    #     elif resolution_string.beginswith(ExperimentNameResolution.ARGUMENT):
    #         pass
    #     else:
    #         # any other..
    #         types = [NewDimensionConflict, ChangedDimensionConflict, MissingDimensionConflict]
    #         for conflict in self.get_remaining_conflicts(types):

    # def _get_potential_renamings(self):
    #     # if new
    #     #     if another is missing:
    #     #         names += [<new name>~><missing name>]
    #     # missing_dim_conflicts = self.branch_builder.get_conflicts([MissingDimensionConflict])
    #     # missing_names = [conflict.dimension.name.strip("/")
    #     #                  for conflict in missing_dim_conflicts
    #     #                  if not conflict.is_resolved]
    #     # new_dim_conflicts = self.branch_builder.get_conflicts([NewDimensionConflict])
    #     # new_names = [conflict.dimension.name.lstrip("/") for conflict in new_dim_conflicts
    #     #              if not conflict.is_resolved]

    #     # names = []
    #     # for missing_name in missing_names:
    #     #     for new_name in new_names:
    #     #         names.append(
    #                   "{}{}{}".format(missing_name, RenameDimensionResolution.MARKER, new_name))
    #     missing_dim_conflicts = self.branch_builder.get_remaining_conflicts(
    #         [MissingDimensionConflict])
    #     new_dim_conflicts = self.branch_builder.get_remaining_conflicts(
    #         [NewDimensionConflict])

    #     names = []
    #     for missing_dim_conflict in missing_dim_conflicts:
    #         for new_dim_conflict in new_dim_conflicts:
    #             resolution = missing_dim_conflict.try_resolve(new_dim_conflict)
    #             names.append(str(resolution))
    #             resolution.revert()
    #
    #     return names

    # def _get_potential_additions(self):
    #     new_dim_conflicts = self.branch_builder.get_remaining_conflicts(
    #         [NewDimensionConflict])
    #     names = []
    #     for conflict in new_dim_conflicts:
    #         resolution = conflict.try_resolve()
    #         names.append(str(resolution))
    #         names.append(str(resolution).rstrip(")") + ", default_value=")
    #         resolution.revert()

    #     return names

    # def _get_potential_changes(self):
    #     changed_dim_conflicts = self.branch_builder.get_remaining_conflicts(
    #         [ChangedDimensionConflict])
    #     names = []
    #     for conflict in changed_dim_conflicts:
    #         resolution = conflict.try_resolve()
    #         names.append(str(resolution))
    #         resolution.revert()

    #     return names

    # def _get_potential_removes(self):
    #     missing_dim_conflicts = self.branch_builder.get_remaining_conflicts(
    #         [MissingDimensionConflict])
    #     names = []
    #     for conflict in missing_dim_conflicts:
    #         resolution = conflict.try_resolve()
    #         names.append(str(resolution))
    #         names.append(str(resolution).rstrip(")") + "'")
    #         resolution.revert()

    #     return names

    # def _get_potential_algo(self):
    #     algo_conflicts = self.branch_builder.get_remaining_conflicts([AlgorithmConflict])
    #     names = []
    #     for conflict in algo_conflicts:
    #         resolution = conflict.try_resolve()
    #         names.append(str(resolution))
    #         resolution.revert()

    #     return names

    # def _get_potential_codes(self):
    #     code_conflicts = self.branch_builder.get_remaining_conflicts([CodeConflict])
    #     names = []
    #     for conflict in code_conflicts:
    #         for change_type in evc.adapters.CodeChange.change_types:
    #             resolution = conflict.try_resolve(change_type)
    #             names.append(str(resolution))
    #             resolution.revert()

    #     return names

    # def _get_potential_branch_name(self):
    #     exp_name_conflicts = self.branch_builder.get_remaining_conflicts([ExperimentNameConflict])
    #     if not exp_name_conflicts:
    #         return []

    #     query = {'refers.root_id': self.branch_builder.experiment_config['refers']['root_id'],
    #              'metadata.user': self.branch_builder.experiment_config['metadata']['user']}
    #     # names = [experiment['name'] for experiment in Database().read('experiments', query)]
    #     names = ["--branch {}".format(experiment['name'])
    #              for experiment in Database().read('experiments', query)]

    #     return names
    #     # if name
    #     #     names += ['--branch <names>']

    # def get_potential_resolution_strings(self):
    #     return (self._get_potential_renamings() +
    #             self._get_potential_additions() +
    #             self._get_potential_changes() +
    #             self._get_potential_removes() +
    #             self._get_potential_algo() +
    #             self._get_potential_codes() +
    #             self._get_potential_branch_name())

    # def add_resolution(self, resolution):
    #     if resolution is None:
    #         return

    #     self.resolutions.append(resolution)
    #     self.conflicts += resolution.new_conflicts

    # def try_resolve(self, conflict, *args, **kwargs):
    #     resolution = self.conflicts.try_resolve(conflict, *args, **kwargs)
    #     if resolution:
    #         self.resolutions.add(resolution)
    #     return resolution

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

        self.conflicts.try_resolve(code_conflicts[0], change_type)

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
            raise RuntimeError('No conflict to solve')

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

        Only dimensions with conflict type `MissingDimensionConflict may be added.

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
