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

import orion.core.cli.database_checks as db_check

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    test_db_parser = parser.add_parser('test_db', help='test_db help')

    test_db_parser.add_argument('-c', '--config', type=argparse.FileType('r'),
                                metavar='path-to-config', help="user provided "
                                "orion configuration file")

    test_db_parser.set_defaults(func=main)

    return test_db_parser


def main(args):
    """Run through all checks for database."""
    shared_dict = args

    checks = db_check.config_checks()

    for check in checks:
        print(check.__doc__, end="")
        error, msg = check(shared_dict)
        if error:
            print("Failure")
            print(msg)
            return
        else:
            print("Success")
