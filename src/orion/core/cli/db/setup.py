#!/usr/bin/env python
"""
Module running the setup command
================================

Creates a configurarion file for the database.

"""

import logging
import os

import yaml

import orion.core
from orion.core.io.database import database_factory
from orion.core.utils.terminal import ask_question

log = logging.getLogger(__name__)
SHORT_DESCRIPTION = "Starts the database configuration wizard"
DESCRIPTION = """
This command starts the database configuration wizard and creates a configuration file for the
database.
"""


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    setup_parser = parser.add_parser(
        "setup", help=SHORT_DESCRIPTION, description=DESCRIPTION
    )

    setup_parser.set_defaults(func=main)

    return setup_parser


# pylint: disable = unused-argument
def main(*args):
    """Build a configuration file."""
    default_file = orion.core.DEF_CONFIG_FILES_PATHS[-1]

    if os.path.exists(default_file):
        cancel = ""
        while cancel.strip().lower() not in ["y", "n"]:
            cancel = ask_question(
                f"This will overwrite {default_file}, do you want to proceed? (y/n) ",
                "n",
            )

        if cancel.strip().lower() == "n":
            return

    # Get database type.
    _type = ask_question(
        "Enter the database",
        choice=sorted(database_factory.get_classes().keys()),
        default="mongodb",
        ignore_case=True,
    ).lower()
    # Get database arguments.
    db_class = database_factory.get_classes()[_type]
    db_args = db_class.get_defaults()
    arg_vals = {}
    for arg_name, default_value in sorted(db_args.items()):
        arg_vals[arg_name] = ask_question(
            f"Enter the database {arg_name}: ", default_value
        )

    config = {"database": {"type": _type, **arg_vals}}

    print("Default configuration file will be saved at: ")
    print(default_file)

    dirs = "/".join(default_file.split("/")[:-1])
    os.makedirs(dirs, exist_ok=True)

    with open(default_file, "w", encoding="utf8") as output:
        yaml.dump(config, output, default_flow_style=False)
