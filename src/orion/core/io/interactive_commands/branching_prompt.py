#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI for conflict solving
========================

Launch the interactive prompt and call the related commands

"""

import argparse
import cmd
import functools
import io
import os
import readline
import shlex
import traceback

from orion.algo.space import Dimension
from orion.core.evc import adapters, conflicts
from orion.core.utils.diff import green, red
from orion.storage.base import get_storage

readline.set_completer_delims(" ")


def wrap_autocomplete(f):
    """Wrap autocomplete to catch errors and print stacktrace"""

    @functools.wraps(f)
    def call(self, *args):

        # Don't know why, it gets reset to default ones between 2 auto-completions
        readline.set_completer_delims(" ")

        try:
            completions = f(self, *args)
        except KeyboardInterrupt:
            raise
        except BaseException:
            print(traceback.format_exc())
            completions = []

        return completions

    return call


def parse_command(f):
    """Wrap command methods to automatically parse using parsers and catch errors to print
    stacktrace without leaving the prompt
    """

    @functools.wraps(f)
    def call(self, arg):

        try:
            options = self.parser.parse_args([f.__name__[3:]] + shlex.split(arg))
        except SystemExit:
            return None

        try:
            rval = f(self, options)
        except KeyboardInterrupt:
            raise
        except BaseException:
            print(traceback.format_exc())
            rval = None

        return rval

    return call


class BranchingPrompt(cmd.Cmd):
    """
    Interactive command prompt to solve the configurations conflicts
    between the parent configuration and the new one.
    """

    intro = (
        "\n\n"
        "Welcome to Orion's experiment branching interactive conflicts resolver\n"
        "-----------------------------------------------------------------------\n\n"
        "If you are unfamiliar with this process, you can type "
        "`help` to print the help message. You can also type `abort` or `(q)uit` at any "
        "moment to quit without saving.\n"
        "\n"
        "%s"
    )
    prompt = "(orion) "

    def __init__(self, branch_builder):
        """Retrieve the instance of ExperimentBranchBuilder containing the conflicts"""
        cmd.Cmd.__init__(self)
        self.branch_builder = branch_builder
        self.abort = False
        self._build_parsers()

    def _build_parsers(self):
        self.parser = argparse.ArgumentParser(prog="(orion)", add_help=False)
        subparsers = self.parser.add_subparsers(title="commands")
        for command in self.get_commands():
            subparser_builder = getattr(self, "_add_{}_parser".format(command), None)
            if subparser_builder:
                subparser_builder(subparsers)
            else:
                self._add_default_subparser(command, subparsers)

    def _build_argument_parser(self, subparsers, name, **kwargs):
        return subparsers.add_parser(name, **kwargs)

    def _add_default_subparser(self, name, subparsers):
        method = getattr(self, "do_{}".format(name))
        subparser = self._build_argument_parser(
            subparsers,
            name,
            help=getattr(method, "__doc__", None),
            description=getattr(method, "__doc__", None),
        )
        return subparser

    def _add_name_parser(self, subparsers):
        subparser = self._build_argument_parser(
            subparsers, "name", help=self.do_name.__doc__
        )
        subparser.add_argument("experiment_name", metavar="experiment-name")
        return subparser

    def _add_code_parser(self, subparsers):
        subparser = self._build_argument_parser(
            subparsers, "code", help=self.do_code.__doc__
        )
        subparser.add_argument("change_type", choices=adapters.CodeChange.types)
        return subparser

    def _add_commandline_parser(self, subparsers):
        subparser = self._build_argument_parser(
            subparsers, "commandline", help=self.do_commandline.__doc__
        )
        subparser.add_argument("change_type", choices=adapters.CommandLineChange.types)
        return subparser

    def _add_config_parser(self, subparsers):
        subparser = self._build_argument_parser(
            subparsers, "config", help=self.do_config.__doc__
        )
        subparser.add_argument("change_type", choices=adapters.ScriptConfigChange.types)
        return subparser

    def _add_add_parser(self, subparsers):
        subparser = self._build_argument_parser(
            subparsers, "add", help=self.do_add.__doc__
        )
        subparser.add_argument("dimension_name", metavar="dimension-name")
        subparser.add_argument("--default-value", default=Dimension.NO_DEFAULT_VALUE)
        return subparser

    def _add_remove_parser(self, subparsers):
        subparser = self._build_argument_parser(
            subparsers, "remove", help=self.do_remove.__doc__
        )
        subparser.add_argument("dimension_name", metavar="dimension-name")
        subparser.add_argument("--default-value", default=Dimension.NO_DEFAULT_VALUE)
        return subparser

    def _add_rename_parser(self, subparsers):
        subparser = self._build_argument_parser(
            subparsers, "rename", help=self.do_rename.__doc__
        )
        subparser.add_argument("old_name", metavar="old-name")
        subparser.add_argument("new_name", metavar="new-name")
        return subparser

    def _add_reset_parser(self, subparsers):
        subparser = self._build_argument_parser(
            subparsers, "reset", help=self.do_reset.__doc__
        )
        subparser.add_argument("resolutions", nargs="+", metavar="resolutions")
        return subparser

    def solve_conflicts(self):
        """Start the prompt for the interactive conflicts solving"""
        self.cmdloop()

    def get_commands(self):
        """Get command names of the prompt"""
        # Get rid of 'do_'
        commands = [
            attr[3:]
            for attr in dir(self)
            if callable(getattr(self, attr)) and attr.startswith("do_")
        ]

        return commands

    def cmdloop(self):
        """Start cmdloop of the prompt"""
        intro = self.intro % self.get_status()
        super(BranchingPrompt, self).cmdloop(intro)

    def get_status(self, options=None):
        """Return a string representing the status"""
        unsolved_conflicts = self.branch_builder.conflicts.get_remaining()
        resolved_conflicts = self.branch_builder.conflicts.get_resolved()

        output = io.StringIO()
        if resolved_conflicts:
            resolution_strings = set(
                str(conflict.resolution) for conflict in resolved_conflicts
            )
            print("Resolutions:", file=output)
            print(file=output)
            for resolution_string in resolution_strings:
                print("    ", green(str(resolution_string)), file=output)
            print(file=output)

            if unsolved_conflicts:
                print(file=output)

        if unsolved_conflicts:

            print("Remaining conflicts:", file=output)
            print(file=output)
            for conflict in unsolved_conflicts:
                print("    ", red(str(conflict)), file=output)
            print(file=output)
        else:
            print(file=output)
            print("Hooray, there is no more conflicts!", file=output)
            print(
                "You can enter 'commit' to leave this prompt and register the new branch",
                file=output,
            )
            print(file=output)

        return output.getvalue()

    # Commands
    @parse_command
    def do_help(self, options):
        """Print help message for all commands"""
        output = io.StringIO(initial_value="", newline="\n")
        self.parser.print_help(file=output)
        print(output.getvalue())

    @parse_command
    def do_h(self, options):
        """Alias for help"""
        self.do_help("")

    def do_shell(self, line):
        """Run a shell command. Ex: (orion) ! pwd"""
        print("running shell command:", line)
        print(os.popen(line).read())

    @parse_command
    def do_status(self, options):
        """Display the current status of the conflicting configuration"""
        print()
        print(self.get_status(options))

    @parse_command
    def do_diff(self, options):
        """Print diff for all conflicts"""
        for conflict in self.branch_builder.conflicts.get():
            if conflict.diff:
                print(conflict)
                print("-" * len(str(conflict)))
                print(conflict.diff)
                print()

    @parse_command
    def do_auto(self, options):
        """Automatically solve conflicts when no feedback from user is necessary."""
        self.branch_builder.manual_resolution = False
        self.branch_builder.resolve_conflicts()

        return True

    @parse_command
    def do_name(self, options):
        """Change the name of the experiment"""
        self.branch_builder.change_experiment_name(options.experiment_name)

        print(
            "TIP: You can use the '-b' or '--branch' command-line "
            "argument to automate the naming process."
        )

        self.do_status("")

    @wrap_autocomplete
    def complete_name(self, text, line, begidx, endidx):
        """Auto-completion of experiment name based on names in the EVC three."""
        if len(line.split(" ")) >= 3:
            names = []
        else:
            query = {
                "refers.root_id": self.branch_builder.experiment_config["refers"][
                    "root_id"
                ],
                "metadata.user": self.branch_builder.experiment_config["metadata"][
                    "user"
                ],
            }
            names = [
                experiment["name"]
                for experiment in get_storage().fetch_experiments(query)
            ]

        return self._get_completions(names, text)

    def _get_completions(self, names, text, ignore=()):
        return [f + " " for f in names if f.startswith(text) and f not in ignore]

    @parse_command
    def do_code(self, options):
        """Set the type of the code change"""
        self.branch_builder.set_code_change_type(options.change_type)

        self.do_status("")

    @wrap_autocomplete
    def complete_code(self, text, line, begidx, endidx):
        """Auto-complete based on possible code change types"""
        names = adapters.CodeChange.types
        if len(line.split(" ")) > 2:
            names = []
        return self._get_completions(names, text)

    @parse_command
    def do_commandline(self, options):
        """Set the type of the commandline change"""
        self.branch_builder.set_cli_change_type(options.change_type)
        self.do_status("")

    @wrap_autocomplete
    def complete_commandline(self, text, line, begidx, endidx):
        """Auto-complete based on possible commandline change types"""
        names = adapters.CommandLineChange.types
        if len(line.split(" ")) > 2:
            names = []
        return self._get_completions(names, text)

    @parse_command
    def do_config(self, options):
        """Set the type of the commandline change"""
        self.branch_builder.set_script_config_change_type(options.change_type)
        self.do_status("")

    @wrap_autocomplete
    def complete_config(self, text, line, begidx, endidx):
        """Auto-complete based on possible script config change types"""
        names = adapters.ScriptConfigChange.types
        if len(line.split(" ")) > 2:
            names = []
        return self._get_completions(names, text)

    @parse_command
    def do_algo(self, options):
        """Resolve the algorithm conflict"""
        self.branch_builder.set_algo()
        self.do_status("")

    @parse_command
    def do_orion_version(self, options):
        """Resolve the orion version conflict"""
        self.branch_builder.set_orion_version()
        self.do_status("")

    @parse_command
    def do_add(self, options):
        """Add the given `new` or `changed` dimension to the configuration"""
        print(
            "TIP: You can use the '~+' marker in place of the usual ~ with "
            "the command-line to solve this conflict automatically."
            "\nEx: -x~+uniform(0,1)"
        )

        self.branch_builder.add_dimension(options.dimension_name, options.default_value)

        self.do_status("")

    @wrap_autocomplete
    def complete_add(self, text, line, begidx, endidx):
        """Auto-complete addition of new or changed dimensions"""
        return self._complete_dim(
            [conflicts.NewDimensionConflict, conflicts.ChangedDimensionConflict],
            text,
            line,
            begidx,
            endidx,
        )

    def _complete_add_new_or_missing(self, conflict, text, line, begidx, endidx):
        """Auto-complete addition of new dimensions or deletion of missing dimensions"""
        dim_is_categorical = conflict.dimension.type == "categorical"

        if len(line.split(" ")) == 3:
            return ["--default-value "]
        elif len(line.split(" ")) == 4 and dim_is_categorical:
            return self._get_completions(
                [repr(f) for f in conflict.dimension.categories], text
            )

        return []

    def _complete_add_changed(self, conflict, text, line, begidx, endidx):
        """Auto-complete addition of changed dimensions"""
        return []

    @parse_command
    def do_remove(self, options):
        """Remove the given `missing` dimension from the configuration"""
        print(
            "TIP: You can use the '~-' marker in place of the usual ~ with "
            "the command-line to solve this conflict automatically."
            "\nEx: -x~-"
        )

        self.branch_builder.remove_dimension(
            options.dimension_name, options.default_value
        )

        self.do_status("")

    @wrap_autocomplete
    def complete_remove(self, text, line, begidx, endidx):
        """Auto-complete deletion of missing dimensions"""
        return self._complete_dim(
            [conflicts.MissingDimensionConflict], text, line, begidx, endidx
        )

    def _complete_dim(self, conflict_types, text, line, begidx, endidx):
        """Auto-complete addition or deletion of new, changed or missing dimensions"""
        potential_conflicts = self.branch_builder.conflicts.get_remaining(
            conflict_types
        )

        names = [
            conflict.dimension.name.lstrip("/") for conflict in potential_conflicts
        ]

        if len(line.split(" ")) == 2:
            return self._get_completions(names, text)

        # Dimension name is fully specified in command line
        names = [name for name in names if name == line.split(" ")[1]]
        if not names:
            return []
        name = names[0]

        potential_conflicts = [
            conflict
            for conflict in potential_conflicts
            if conflict.dimension.name.lstrip("/") == name
        ]
        if not potential_conflicts:
            return []
        conflict = potential_conflicts[0]

        if isinstance(
            conflict,
            (conflicts.NewDimensionConflict, conflicts.MissingDimensionConflict),
        ):
            return self._complete_add_new_or_missing(
                conflict, text, line, begidx, endidx
            )
        else:
            return self._complete_add_changed(conflict, text, line, begidx, endidx)

    @parse_command
    def do_rename(self, options):
        """Rename a dimension"""
        print(
            "TIP: You can use the '~>' marker in place of the usual ~ with "
            "the command-line to solve this dimension automatically."
            "\nEx: -x~>y"
        )
        self.branch_builder.rename_dimension(options.old_name, options.new_name)

        self.do_status("")

    @wrap_autocomplete
    def complete_rename(self, text, line, begidx, endidx):
        """Auto-complete rename based on non-resolved missing dimensions and new dimensions
        conflicts
        """
        if len(line.split(" ")) < 3:
            potential_conflicts = self.branch_builder.conflicts.get_remaining(
                [conflicts.MissingDimensionConflict]
            )
        elif len(line.split(" ")) == 3:
            potential_conflicts = self.branch_builder.conflicts.get_remaining(
                [conflicts.NewDimensionConflict]
            )
        else:
            potential_conflicts = []

        names = [
            conflict.dimension.name.lstrip("/") for conflict in potential_conflicts
        ]

        return self._get_completions(names, text)

    @parse_command
    def do_reset(self, options):
        """Revert a resolution and mark the conflict as non-resolved"""
        for resolution in options.resolutions:
            self.branch_builder.reset(resolution.strip("' ").replace('"', "'"))

        self.do_status("")

    @wrap_autocomplete
    def complete_reset(self, text, line, begidx, endidx):
        """Auto-complete reset based on current resolutions"""
        names = list(map(str, self.branch_builder.conflicts.get_resolutions()))
        names = [shlex.quote(name) for name in names]
        try:
            ignore = [shlex.quote(t) for t in shlex.split(line)]
        except ValueError:
            ignore = [shlex.quote(t) for t in shlex.split(line + "'")]

        names = [name.replace("'\"'\"'", '"') for name in names]

        return self._get_completions(names, text, ignore=ignore)

    @parse_command
    def do_abort(self, options):
        """Exit the prompt without saving"""
        print("Closing interactive conflicts solver. Experiment branch won't be saved.")

        self.abort = True
        return True

    @parse_command
    def do_quit(self, options):
        """Exit the prompt without saving"""
        return self.do_abort("")

    @parse_command
    def do_q(self, options):
        """Alias for quit"""
        return self.do_quit("")

    @parse_command
    def do_commit(self, options):
        """Exit the prompt and creates the adapters inside the builders"""
        if not self.branch_builder.is_resolved:
            print(
                "Error: There is still remaining issues. Enter 'abort' or 'quit' to leave "
                "without registering the new experiment branch."
            )
            return False

        new_name = self.branch_builder.conflicting_config["name"]
        print("Registering experiment branch '{0}'.".format(new_name))

        return True
