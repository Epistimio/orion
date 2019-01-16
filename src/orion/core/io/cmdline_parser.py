# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.cmdline_parser` -- Parse command line arguments for Orion
=============================================================================

.. module:: cmdline_parser
   :platform: Unix
   :synopsis: Parsing and building of the command line input for using script

Simplify the parsing of a command line by storing every values inside an `OrderedDict`
mapping the name of the argument to its value as a key-pair relation. Positional arguments
are stored in the format `_X` where `X` represent the index of that argument inside
the command line.

CmdlineParser provides an interface to parse command line arguments from input but also
facilities to build it again as a list or an already formatted string.
"""
from collections import OrderedDict
import os


class CmdlineParser(object):
    """Simple class for commandline arguments parsing.

    CmdlineParser aims at providing a simple class to interpret commandline arguments
    for the purposes of Orion. It can transform a list of string representing arguments to their
    corresponding values. It can also recreate that string from the values by maintaing a template
    of the way the arguments were passed.

    Attributes
    ----------
    arguments : OrderedDict
        Commandline arguments' name and value(s).
    template : list
        List of template-ready strings (in the form 'something_{0}') for formatting.

    """

    def __init__(self):
        """See `CmdlineParser` description"""
        self.arguments = OrderedDict()

        # TODO Handle parsing twice.
        self._already_parsed = False
        self.template = []

    def format(self, configuration):
        """Format the current template.

        Recreate the string of argument using the template made at parse-time and
        the values inside the `configuration` argument.

        Parameters
        ----------
        configuration : dict
            Dictionary storing the keys and values to be passed to the `format` function.
            This would typically be `parser.arguments` where `parser` is a `CmdlineParser`
            instance.

        Returns
        -------
        str
            A recreated string of the commandline passed to Orion.

        """
        return " ".join(self.template).format(**configuration)

    def parse(self, commandline):
        """Parse the `commandline` argument.

        Create an OrderedDict where the keys are the names of the arguments and the values
        are the actual values of the each argument and a template to be formatted later by the
        user to recover the original commandline with priors replaced by value. The arguments
        can be a single value or a list of values. This also supports positional arguments.

        Parameters
        ----------
        commandline : list
            List of string representing the commmandline arguments.

        Returns
        -------
        OrderedDict
            Dictionary holding the values of every argument. The keys are the arguments' name.

        Raises
        ------
        ValueError
            If there is a duplicate argument

        Notes
        -----
        By default, all values are `str` unless their types have been changed in the meantime.
        Unnamed arguments's key follow the format '_X' where `X` is the index of that argument
        in the list. For example, `val1 val2` would be parsed as
        `{'_pos_0': 'val1', '_pos_1': 'val2}`.

        Arrays are parsed starting at the first named argument until the next one. For example :
        `--arg1 value1 value2 --arg2 value3 value4` will be parsed as :
        `{'arg1': ['value1', 'value2'], 'arg2': ['value3', 'value4']}`.

        File paths are extended to their absolute forms.

        If the commandline contains optional arguments before a subcommand, the arguments
        will be wrongly parsed : `somecommand --optional argument subcommand --another argument`
        will be parsed as `{'optional': ['argument', 'subcommand'], 'another': 'argument'}`

        Warning
        -------
        When dealing with an empty list (of the form `--args`), the value is a boolean.

        """
        if not commandline:
            return self.arguments

        self.arguments = OrderedDict()
        self._parse_arguments(commandline)

        for key, value in self.arguments.items():
            # Handle positional arguments
            if key.startswith("_"):
                self.template.append("{" + key + "}")

            # Handle optional ones
            else:
                key_name = self._key_to_arg(key)

                if key_name in self.template:
                    continue

                self.template.append(key_name)

                # Ignore value as key is a boolean argument
                if isinstance(value, bool):
                    continue

                if not isinstance(value, list):
                    template = "{" + key + "}"
                    self.template.append(template)
                    continue

                for pos in range(len(value)):
                    template = "{" + key + "[" + str(pos) + "]}"
                    self.template.append(template)

        self._already_parsed = True

        return self.arguments

    def _key_to_arg(self, key):
        arg = key.replace(self._underscore_token, "_").replace(self._dash_token, "-")

        if len(arg) > 1:
            return "--" + arg

        return "-" + arg

    def _parse_arguments(self, arguments):
        argument_name = None

        for arg in arguments:
            # Handle keyworded arguments
            if arg.startswith("-"):
                # Recover the argument name
                arg_parts = arg.split("=")
                argument_name = self._arg_to_key(arg)

                # Make sure we're not defining the same argument twice
                if argument_name in self.arguments.keys():
                    raise ValueError("Two arguments have the same name: {}".format(argument_name))

                self.arguments[argument_name] = []

                if len(arg_parts) > 1 and "=".join(arg_parts[1:]).strip(" "):
                    self.arguments[argument_name].append("=".join(arg_parts[1:]))

            # If the argument did not start with `-` but we have an argument name
            # That means that this value belongs to that argument name list
            elif argument_name is not None and arg.strip(" "):
                self.arguments[argument_name].append(arg)

            # No argument name means we have not reached them yet, so we're still in the
            # Positional arguments part
            elif argument_name is None:
                self.arguments["_pos_{}".format(len(self.arguments))] = arg

        for key, value in self.arguments.items():
            if not value:
                value = True
            elif isinstance(value, list) and len(value) == 1:
                value = value[0]

            value = self._parse_paths(value)

            self.arguments[key] = value

    def _arg_to_key(self, full_arg):
        arg_parts = full_arg.split("=")
        arg = arg_parts[0]

        if arg.startswith("--") and len(arg) == 3:
            raise ValueError(
                "Arguments with two dashes should have more than one letter: {}".format(arg))

        elif not arg.startswith("--") and arg.startswith("-") and len(arg) > 2:
            raise ValueError(
                "Arguments with one dashes should have only one letter: {}".format(arg))

        return arg.lstrip("-").replace("_", self._underscore_token) \
                              .replace("-", self._dash_token)

    def _parse_paths(self, value):
        if isinstance(value, list):
            return [self._parse_paths(item) for item in value]

        if isinstance(value, str) and os.path.exists(value):
            return os.path.abspath(value)

        return value
