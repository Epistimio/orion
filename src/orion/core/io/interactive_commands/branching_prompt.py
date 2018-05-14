#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.interactive_commands.base_commands.py` -- Module hosting
the base commands of the interactive prompt for branch conflicts solving as
well as handling the dispatch to specific sub commands classes.
============================================================================
.. module:: base_commands
    :platform: Unix
    :synopsis: Launch the interactive prompt and call the related commands
"""

import cmd


class BranchingPrompt(cmd.Cmd):
    intro = 'Welcome to Orion\'s experiment branching interactive conflicts '
    'solver. If you are unfamiliar with this process, you can type '
    '`help` to open the help menu. You can also type `abort` at any'
    'moment to quit without saving.'
    prompt = '(orion) '

    conflicts_message = {'new': 'Dimension {} is new',
                         'changed': 'Dimension {} has changed from {} to {}',
                         'missing': 'Dimension {} is missing',
                         'experiment': 'Experiment name {0} is conflicting'}

    def __init__(self, branch_builder):
        cmd.Cmd.__init__(self)
        self.branch_builder = branch_builder

    def solve_conflicts(self):
        """Starts the prompt for the interactive conflicts solving"""
        self.cmdloop()

    def do_status(self, arg):
        'Display the current status of the conflicting configuration'

        if len(arg) == 0:
            self._print_conflicts_message('Solved conflicts', self.branch_builder.solved_conflicts)
            self._print_conflicts_message('Unsolved conflicts', self.branch_builder.conflicts)
        else:
            self._print_dimension_status(arg)

    def _print_dimension_status(self, name):
        dimension_status, key, value = self.branch_builder.get_dimension_status(name)

        if key is None or value is None:
            print('Invalid dimension name')
            return

        if dimension_status == False:
            print('Unsolved')
        else:
            print('Solved')
        
        if key != 'changed':
            print(self.conflicts_message[key].format(value.name))
            print(value)
        else:
            print(self.conflicts_message[key].format(value.name, self.branch_builder.get_old_dimension_value(value.name), value))
    
    def _print_conflicts_message(self, preprint_message, conflicts_dict):
        if any(conflicts_dict.values()):
            print(preprint_message)
            for k in conflicts_dict:
                if k == 'experiment' and conflicts_dict[k] != '':
                    print(self.conflicts_message[k].format(conflicts_dict[k]))
                else:
                    for v in conflicts_dict[k]:
                        if k != 'changed':
                            print(self.conflicts_message[k].format(v.name))
                        else:
                            print(self.conflicts_message[k].format(v.name, self.branch_builder.get_old_dimension_value(v.name), v))

    def do_name(self, arg):
        'Change the name of the experiment'
        arg = arg.split(' ')[0]

        if len(arg) > 0:
            self.branch_builder.change_experiment_name(arg)
        else:
            print('Invalid experiment name')

    def do_add(self, arg):
        'Add a new hyperparameter to the experiment'
        name = arg.split(' ')[0]
        try:
            self.branch_builder.add_dimension(name)
        except IndexError as ex:
            print('Invalid dimension name')
            return

    def do_exit(self, arg):
        print('Closing interactive conflicts solver')
        return True
