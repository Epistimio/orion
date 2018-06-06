# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli` -- Base class and function utilities for cli
==================================================================

.. module:: cli
   :platform: Unix
   :synopsis: Orion main parser class and helper functions to parse command-line options

"""
import argparse
import logging
import textwrap

import orion


class OrionArgsParser:
    """Parser object handling the upper-level parsing of Oríon's arguments."""

    def __init__(self, description):
        """Create the pre-command arguments"""
        self.description = description

        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent(description))

        self.parser.add_argument(
            '-V', '--version',
            action='version', version='orion ' + orion.core.__version__)

        self.parser.add_argument(
            '-v', '--verbose',
            action='count', default=0,
            help="logging levels of information about the process (-v: INFO. -vv: DEBUG)")

        self.subparsers = self.parser.add_subparsers(help='sub-command help')

    def get_subparsers(self):
        """Return the subparser object for this parser."""
        return self.subparsers

    def parse(self, argv):
        """Call argparse and generate a dictionary of arguments' value"""
        args = vars(self.parser.parse_args(argv))

        verbose = args.pop('verbose', 0)
        if verbose == 1:
            logging.basicConfig(level=logging.INFO)
        elif verbose == 2:
            logging.basicConfig(level=logging.DEBUG)

        function = args.pop('func')
        return args, function

    def execute(self, argv):
        """Execute main function of the subparser"""
        args, function = self.parse(argv)
        function(args)


def get_basic_args_group(parser):
    """Return the basic arguments for any command."""
    basic_args_group = parser.add_argument_group(
        "Oríon arguments (optional)",
        description="These arguments determine orion's behaviour")

    basic_args_group.add_argument(
        '-n', '--name',
        type=str, metavar='stringID',
        help="experiment's unique name; "
             "(default: None - specified either here or in a config)")

    basic_args_group.add_argument('-c', '--config', type=argparse.FileType('r'),
                                  metavar='path-to-config', help="user provided "
                                  "orion configuration file")

    return basic_args_group


def get_user_args_group(parser):
    """
    Return the user group arguments for any command.
    User group arguments are composed of the user script and the user args
    """
    usergroup = parser.add_argument_group(
        "User script related arguments",
        description="These arguments determine user's script behaviour "
                    "and they can serve as orion's parameter declaration.")

    usergroup.add_argument(
        'user_script', type=str, metavar='path-to-script',
        help="your experiment's script")

    usergroup.add_argument(
        'user_args', nargs=argparse.REMAINDER, metavar='...',
        help="Command line arguments to your script (if any). A configuration "
             "file intended to be used with 'userscript' must be given as a path "
             "in the **first positional** argument OR using `--config=<path>` "
             "keyword argument.")

    return usergroup
