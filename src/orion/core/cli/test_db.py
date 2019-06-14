#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.test_db` -- Module to check if the DB worrks
=================================================================

.. module:: test_db
   :platform: Unix
   :synopsis: Gets an experiment and iterates over it until one of the exit conditions is met

"""
import logging
import types

from orion.core.cli import base as cli
import orion.core.cli.database_checks as db_check

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    test_db_parser = parser.add_parser('test_db', help='test_db help')

    cli.get_basic_args_group(test_db_parser)

    test_db_parser.set_defaults(func=main)

    return test_db_parser


def main(args):
    """Run through all checks for database."""
    shared_dict = args
    checks = [getattr(db_check, func) for func in dir(db_check)
              if isinstance(getattr(db_check, func), types.FunctionType)]

    checks = sorted(checks, key=lambda x: hex(id(x)))

    for check in checks:
        print(check.__doc__, end="")
        error, msg = check(shared_dict)
        if error:
            print("Failure")
            print(msg)
            return
        else:
            print("Success")
