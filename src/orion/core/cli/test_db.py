#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.test_db` -- Module to check if the DB worrks
=================================================================

.. module:: test_db
   :platform: Unix
   :synopsis: Runs multiple checks to see if the database was correctly setup.

"""
import argparse
import logging

from orion.core.cli.checks.creation import CreationStage
from orion.core.cli.checks.operations import OperationsStage
from orion.core.cli.checks.presence import PresenceStage
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.utils.exceptions import CheckError

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    test_db_parser = parser.add_parser('test-db', help='test_db help')

    test_db_parser.add_argument('-c', '--config', type=argparse.FileType('r'),
                                metavar='path-to-config', help="user provided "
                                "orion configuration file")

    test_db_parser.set_defaults(func=main)

    return test_db_parser


def main(args):
    """Run through all checks for database."""
    experiment_builder = ExperimentBuilder()
    presence_stage = PresenceStage(experiment_builder, args)
    creation_stage = CreationStage(presence_stage)
    operations_stage = OperationsStage(creation_stage)
    stages = [presence_stage, creation_stage, operations_stage]

    try:
        for stage in stages:
            for check in stage.checks():
                print(check.__doc__, end='.. ')
                status, msg = check()
                print(status)

                if status == "Skipping":
                    print(msg)

            stage.post_stage()

    except CheckError as ex:
        print("Failure")
        print(ex)
