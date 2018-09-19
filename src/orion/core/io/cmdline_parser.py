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
    def __init__(self):
        self.arguments = OrderedDict()
        self._already_parsed = False
        self.template = []

    def parse(self, commandline):
        if not commandline:
            return self.arguments

        self.parse_arguments(commandline)

        for key, value in self.arguments.items():
            # TODO: Support passing the same commandline but slightly different
            # Handle positional arguments
            if key.startswith("_"):
                if self._already_parsed:
                    raise RuntimeError("Cannot branch using positional arguments.")

                self.template.append("{" + key + "}")

            # Handle optional ones
            else:
                key_name = self.key_to_arg(key)

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

                for pos, item in enumerate(value):
                    template = "{" + key + "[" + str(pos) + "]}"
                    self.template.append(template)

        self._already_parsed = True

        return self.configuration

    def key_to_arg(self, key):
        arg = key.replace("!!", "_").replace("??", "-")

        if len(arg) > 1:
            return "--" + arg

        return "-" + arg

    def parse_arguments(self, arguments):
        argument_name = None

        for arg in arguments:
            # Handle keyworded arguments
            if arg.startswith("-"):
                # Recover the argument name
                arg_parts = arg.split("=")
                argument_name = self.arg_to_key(arg_parts)

                # Make sure we're not defining the same argument twice
                if argument_name in self.arguments.keys():
                    raise ValueError("Two arguments have the same name: {}".format(argument_name))

                self.arguments[argument_name] = []

                if len(arg_parts) > 1 and "=".join(arg[1:]).strip(" "):
                    self.arguments[argument_name].append("=".join(arg[1:]))

            # If the argument did not start with `-` but we have an argument name
            # That means that this value belongs to that argument name list
            elif argument_name is not None and arg.strip(" "):
                self.arguments[argument_name].append(arg)

            # No argument name means we have reached them yet, so we're still in the
            # Positional arguments part
            elif argument_name is None:
                self.arguments["_pos_{}".format(len(self.arguments))] = arg

        for key, value in self.arguments.items():
            if not value:
                value = True
            elif isinstance(value, list) and len(value) == 1:
                value = value[0]

            value = self.parse_paths(value)

            self.arguments[key] = value

    def arg_to_key(self, arg_parts):
        arg = arg_parts[0]

        if arg.startswith("--") and len(arg) == 3:
            raise ValueError(
                "Arguments with two dashes should have more than one letter: {}".format(arg))

        elif not arg.startswith("--") and arg.startswith("-") and len(arg) > 2:
            raise ValueError(
                "Arguments with one dashes should have only one letter: {}".format(arg))

        return arg.lstrip("-").replace("_", "!!").replace("-", "??")

    def parse_paths(self, value):
        if isinstance(value, list):
            return [self.parse_paths(item) for item in value]

        if isinstance(value, str) and os.path.exists(value):
            return os.path.abspath(value)

        return value
