#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.interactive_commands.base_commands.py` -- CLI for conflict solving
======================================================================================
.. module:: base_commands
    :platform: Unix
    :synopsis: Launch the interactive prompt and call the related commands
"""

import cmd


class BranchingPrompt(cmd.Cmd):
    """
    Interactive command prompt to solve the configurations conflicts
    between the parent configuration and the new one.
    """

    intro = 'Welcome to Orion\'s experiment branching interactive conflicts '
    'solver. If you are unfamiliar with this process, you can type '
    '`help` to open the help menu. You can also type `abort` at any'
    'moment to quit without saving.'
    prompt = '(orion) '

    special_keywords = {'~new': 'new',
                        '~changed': 'changed',
                        '~missing': 'missing'}

    conflicts_message = {'new': 'Dimension {} is new',
                         'changed': 'Dimension {} has changed from {} to {}',
                         'missing': 'Dimension {} is missing',
                         'experiment': 'Experiment name {0} is conflicting'}

    def __init__(self, branch_builder):
        """Retrieve the instance of ExperimentBranchBuilder containing the conflicts"""
        cmd.Cmd.__init__(self)
        self.branch_builder = branch_builder
        self.abort = False

    def solve_conflicts(self):
        """Start the prompt for the interactive conflicts solving"""
        self.cmdloop()

    # Commands
    def do_status(self, arg):
        """Display the current status of the conflicting configuration"""
        if len(arg) == 0:
            def filter_true(c):
                return c.is_solved is True

            def filter_false(c):
                return not filter_true(c)

            solved_conflicts = list(self.branch_builder.filter_conflicts(filter_true))
            conflicts = list(self.branch_builder.filter_conflicts(filter_false))

            if len(solved_conflicts) > 0:
                print('Solved')
                self._print_conflicts_message(solved_conflicts)
            if len(conflicts) > 0:
                if len(solved_conflicts) > 0:
                    print()
                print('Unsolved')
                self._print_conflicts_message(conflicts)

        else:
            self._print_singular_status(arg)

    def do_name(self, arg):
        """Change the name of the experiment"""
        arg = arg.split(' ')[0]

        if len(arg) > 0:
            self.branch_builder.change_experiment_name(arg)
        else:
            print('Invalid experiment name')

    def do_add(self, arg):
        """Add the given `new` or `changed` dimension to the configuration"""
        self._call_function_for_all_args(arg, self.branch_builder.add_dimension)

    def do_remove(self, arg):
        """Remove the given `missing` dimension from the configuration"""
        self._call_function_for_all_args(arg, self.branch_builder.remove_dimension)

    def do_rename(self, arg):
        """
        Usage : rename `old` `new`
        Rename old dimension to new
        """
        args = arg.split(' ')

        if len(args) < 2:
            print('Missing arguments')
            return
        try:
            self.branch_builder.rename_dimension(args)
        except ValueError as ex:
            print(ex)

    def do_reset(self, arg):
        """Mark dimension as unsolved"""
        self._call_function_for_all_args(arg, self.branch_builder.reset_dimension)

    def do_abort(self, arg):
        """Exit the prompt without saving"""
        print('Closing interactive conflicts solver')

        self.abort = True
        return True

    def do_commit(self, arg):
        """Exit the prompt and creates the adapters inside the builders"""
        return True

    def do_commit(self, arg):
        print(self.branch_builder.operations)
        return True

    # Helper functions
    def _call_function_for_all_args(self, arg, function):
        try:
            function(arg)
        except ValueError as ex:
            print('Invalid dimension name {}'.format(arg))

    def _print_dimension_status(self, name):
        conflict = self.branch_builder.get_dimension_conflict(name)

        print('Solved' if conflict.is_solved else 'Unsolved')

        self._print_dimension_conflict_info(conflict)

    def _print_conflicts_message(self, conflicts):
        for conflict in conflicts:
            self._print_singular_conflict_info(conflict)

    def _print_singular_conflict_info(self, conflict):
        if conflict.status != 'changed':
            print(self.conflicts_message[conflict.status].format(conflict.dimension.name))
            print(conflict.dimension)
        else:
            print(self.conflicts_message[conflict.status]
                  .format(conflict.dimension.name,
                  self.branch_builder.get_old_dimension_value(conflict.dimension.name),
                  conflict.dimension))
