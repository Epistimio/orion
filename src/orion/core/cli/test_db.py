#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to check if the DB works
===============================

Runs multiple checks to see if the database was correctly setup.

"""
import argparse
import logging

from orion.core.cli.db.test import main

log = logging.getLogger(__name__)
DESCRIPTION = "(DEPRECATED) Use command `orion db test` instead"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    test_db_parser = parser.add_parser(
        "test-db", help=DESCRIPTION, description=DESCRIPTION
    )

    test_db_parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="user provided " "orion configuration file",
    )

    test_db_parser.set_defaults(func=wrap_main)

    return test_db_parser


def wrap_main(args):
    """Run through all checks for database."""
    log.warning(
        "Command `orion test-db` is deprecated and will be removed in v0.2.0. Use "
        "`orion db test` instead."
    )

    main(args)
