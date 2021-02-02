# -*- coding: utf-8 -*-
"""
Base class and function utilities for cli
=========================================
"""
import argparse
import logging
import sys
import textwrap

import orion
from orion.core.io.database import DatabaseError
from orion.core.utils.exceptions import (
    BranchingEvent,
    InexecutableUserScript,
    MissingResultFile,
    NoConfigurationError,
    NoNameError,
)

CLI_DOC_HEADER = "Oríon CLI for asynchronous distributed optimization"


class OrionArgsParser:
    """Parser object handling the upper-level parsing of Oríon's arguments."""

    def __init__(self, description=CLI_DOC_HEADER):
        """Create the pre-command arguments"""
        self.description = description

        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent(description),
        )

        self.parser.add_argument(
            "-V",
            "--version",
            action="version",
            version="orion " + orion.core.__version__,
        )

        self.parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="logging levels of information about the process (-v: INFO. -vv: DEBUG)",
        )

        self.parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            help="Use debugging mode with EphemeralDB.",
        )

        self.subparsers = self.parser.add_subparsers(dest="command")

    def get_subparsers(self):
        """Return the subparser object for this parser."""
        return self.subparsers

    def parse(self, argv):
        """Call argparse and generate a dictionary of arguments' value"""
        args = vars(self.parser.parse_args(argv))

        verbose = args.pop("verbose", 0)
        levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        logging.basicConfig(
            format="%(asctime)-15s::%(levelname)s::%(name)s::%(message)s",
            level=levels.get(verbose, logging.DEBUG),
        )

        if args["command"] is None:
            self.parser.parse_args(["--help"])

        function = args.pop("func", None)
        empty_command = (argv[-1] if argv else sys.argv[-1]) == args["command"]
        if function is None or (empty_command and args.pop("help_empty", False)):
            self.parser.parse_args([args["command"], "--help"])

        return args, function

    def execute(self, argv):
        """Execute main function of the subparser"""
        try:
            args, function = self.parse(argv)
            returncode = function(args)
        except (
            NoConfigurationError,
            NoNameError,
            DatabaseError,
            MissingResultFile,
            BranchingEvent,
            InexecutableUserScript,
        ) as e:
            print("Error:", e, file=sys.stderr)

            if args.get("verbose", 0) >= 2:
                raise e

            return 1

        except KeyboardInterrupt:
            print("Orion is interrupted.")
            return 130

        return 0 if returncode is None else returncode


def get_basic_args_group(
    parser,
    group_name="Oríon arguments",
    group_help="These arguments determine orion's behaviour",
):
    """Return the basic arguments for any command."""
    basic_args_group = parser.add_argument_group(group_name, description=group_help)

    basic_args_group.add_argument(
        "-n",
        "--name",
        type=str,
        metavar="stringID",
        help="experiment's unique name; "
        "(default: None - specified either here or in a config)",
    )

    basic_args_group.add_argument(
        "-u",
        "--user",
        type=str,
        help="user associated to experiment's unique name; "
        "(default: $USER - can be overriden either here or in a config)",
    )

    basic_args_group.add_argument(
        "-v",
        "--version",
        type=int,
        help="specific version of experiment to fetch; "
        "(default: None - latest experiment.)",
    )

    basic_args_group.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="user provided " "orion configuration file",
    )

    return basic_args_group


def get_user_args_group(parser):
    """
    Return the user group arguments for any command.
    User group arguments are composed of the user script and the user args
    """
    usergroup = parser.add_argument_group(
        "User script related arguments",
        description="These arguments determine user's script behaviour "
        "and they can serve as orion's parameter declaration.",
    )

    usergroup.add_argument(
        "user_args",
        nargs=argparse.REMAINDER,
        metavar="...",
        help="Command line of user script. A configuration "
        "file intended to be used with 'userscript' must be given as a path "
        "in the **first positional** argument OR using `--config=<path>` "
        "keyword argument.",
    )

    return usergroup
