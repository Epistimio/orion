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

    `CmdlineParser` aims at providing a simple interface to interpret commandline arguments
    for a general purpose. This class exposes an interface to transform a list of strings
    representing the commandline in a dictionary of key-value conserving the order of
    the arguments, the value of named arguments, be it boolean, single-valued or multiple ones,
    as well as handling the definition of argument through the `=` sign.

    Attributes
    ----------
    arguments : OrderedDict
        Commandline arguments' name and value(s).
    template : list
        List of template-ready strings for formatting.

    See Also
    --------
    parse : Parse the list of string and defines the form of the template.

    """

    def __init__(self):
        """See `CmdlineParser` description"""
        self.arguments = OrderedDict()

        # TODO Handle parsing twice.
        self._already_parsed = False
        self.template = []

    def format(self, configuration):
        """Format the current template.

        Return a list of string where the arguments have been given the values inside
        the `configuration` argument.

        Parameters
        ----------
        configuration : dict
            Dictionary containing the keys and values for each argument of the commandline.

        Returns
        -------
        list
            A list ressembling the one given to the `parse` method where every argument has a value.

        """
        # It is easier to build the command line and return a list this way
        formatted = []
        for item in self.template:
            if item.startswith('-'):
                formatted.append(item)
            else:
                formatted.append(item.format(**configuration))

        return formatted

    def parse(self, commandline):
        """Parse the `commandline` argument.

        This function will go through the list of strings and create a dictionary
        mapping every argument to its value(s). It will also form a template to recreate such a
        list. If `commandline` is empty or the `parse` function has already been called, it
        will return the dictionary of the already-parsed arguments.

        Parameters
        ----------
        commandline : list
            List of string representing the commmandline arguments.

        Returns
        -------
        OrderedDict
            Dictionary holding the values of every argument. The keys are the name of the arguments.

        Raises
        ------
        ValueError
            This exception is raised then the parser detects a duplicate argument.

        Notes
        -----
        Each argument is stored inside an `OrderedDict` to preserve its position inside the
        commandline string. Inside that dictionary, the keys are created following these rules:

        -If the argument is a positional one, its key will be `'_pos_x'` where `x` is its index
        inside the list.
        -Otherwise, the key will be equal to the name of the argument minus the prefixed dashes.

        The values are stored in the following ways:

        -If the argument is a positional one, its value is simply its value inside the commandline.
        -If the argument is a named, boolean argument, its value is simply `True`.
        -If the argument is a named, single-valued argument, its value is stored as-is.
        -If the argument is a named, multi-valued argument, its value is a list containing each
        value following it until the next named argument.

        Positional arguments following a named argument are not currently supported.

        Examples
        --------

        Positional and named arguments:

        >>> parser = CmdlineParser()
        >>> parser.parse('python 1 --arg value'.split(' '))
        OrderedDict([('_pos_0', 'python'), ('_pos_1', '1'), ('arg', 'value')])

        Named boolean argument:

        >>> parser.parse('python --boolean'.split(' '))
        OrderedDict([('_pos_0', 'python'), ('boolean', True)])

        Named multi-valued argument:

        >>> parser.parse('python --args value1 value2'.split(' '))
        OrderedDict([('_pos_0', 'python'), ('args', ['value1', 'value2'])])

        Named argument defined with `=`:

        >>> parser.parse('python --arg=value'.split(' '))
        OrderedDict([('_pos_0', 'python'), ('arg', 'value')])

        """
        if not commandline or self._already_parsed:
            return self.arguments

        self.arguments = OrderedDict()
        self._parse_arguments(commandline)

        for key, value in self.arguments.items():
            # Handle positional arguments
            if key.startswith("_"):
                self.template.append("{" + key + "}")

            # Handle optional ones
            else:
                arg = self._key_to_arg(key)

                if arg in self.template:
                    continue

                self.template.append(arg)

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

    def _key_to_arg(self, arg):
        if len(arg) > 1:
            return "--" + arg

        return "-" + arg

    def _parse_arguments(self, commandline):
        argument_name = None

        for item in commandline:
            # Handle keyworded arguments
            if item.startswith("-"):
                # Make sure we're not defining the same argument twice
                argument_name = item.lstrip('-')
                # If the argument is in the form of `--name=value`
                argument_parts = argument_name.split('=')
                argument_name = argument_parts[0]

                if argument_name in self.arguments.keys():
                    raise ValueError("Conflict: two arguments have the same name: {}"
                                     .format(argument_name))

                self.arguments[argument_name] = []

                if len(argument_parts) > 1:
                    self.arguments[argument_name].append(argument_parts[-1])

            # If the argument did not start with `-` but we have an argument name
            # That means that this value belongs to that argument name list
            elif argument_name is not None and item.strip(" "):
                self.arguments[argument_name].append(item)

            # No argument name means we have not reached them yet, so we're still in the
            # Positional arguments part
            elif argument_name is None:
                self.arguments["_pos_{}".format(len(self.arguments))] = item

        for key, value in self.arguments.items():
            # Loop through the items and check if their value is a list
            # If it is, and the length is 0, that means it is a boolean args.
            # If its value is 1, it only has a single element and we unpack it.
            if isinstance(value, list):
                if not len(value):
                    value = True
                elif len(value) == 1:
                    value = value[0]

            value = self._parse_paths(value)
            self.arguments[key] = value

    def _parse_paths(self, value):
        if isinstance(value, list):
            return [self._parse_paths(item) for item in value]

        if isinstance(value, str) and os.path.exists(value):
            return os.path.abspath(value)

        return value
