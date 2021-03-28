#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to check if the DB works
===============================

Runs multiple checks to see if the database was correctly setup.

"""
import argparse
import logging

from orion.core.cli.checks.creation import CreationStage
from orion.core.cli.checks.operations import OperationsStage
from orion.core.cli.checks.presence import PresenceStage
from orion.core.utils.exceptions import CheckError

log = logging.getLogger(__name__)
SHORT_DESCRIPTION = "Verifies that the database is correctly configured"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    test_db_parser = parser.add_parser(
        "test", help=SHORT_DESCRIPTION, description=SHORT_DESCRIPTION
    )

    test_db_parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="user provided " "orion configuration file",
    )

    test_db_parser.set_defaults(func=main)

    return test_db_parser


def main(args):
    """Run through all checks for database."""
    presence_stage = PresenceStage(args)
    creation_stage = CreationStage(presence_stage)
    operations_stage = OperationsStage(creation_stage)
    stages = [presence_stage, creation_stage, operations_stage]

    for stage in stages:
        for check in stage.checks():
            print(check.__doc__, end=".. ")
            try:
                status, msg = check()
                print(status)
                if status == "Skipping":
                    print(msg)
            except CheckError as ex:
                print("Failure")
                print(ex)

        stage.post_stage()
