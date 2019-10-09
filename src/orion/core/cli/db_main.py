#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.test_db` -- Module to check if the DB worrks
=================================================================

.. module:: test_db
   :platform: Unix
   :synopsis: Runs multiple checks to see if the database was correctly setup.

"""
import logging

from orion.core.utils import module_import

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparsers that needs to be used for this command"""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help

    db_parser = parser.add_parser('db', help='test_db help')
    subparsers = db_parser.add_subparsers(help='sub-command help')

    load_modules_parser(subparsers)

    return db_parser


def load_modules_parser(subparsers):
    """Search through the `cli.db` folder for any module containing a `get_parser` function"""
    modules = module_import.load_modules_in_path('orion.core.cli.db',
                                                 lambda m: hasattr(m, 'add_subparser'))

    for module in modules:
        get_parser = getattr(module, 'add_subparser')
        get_parser(subparsers)
