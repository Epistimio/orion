#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli` -- Functions that define console scripts
================================================================

.. module:: cli
   :platform: Unix
   :synopsis: Helper functions to setup an experiment and execute it.

"""
import logging

from orion.core.cli.base import OrionArgsParser
import orion.core.utils.module_import

log = logging.getLogger(__name__)

CLI_DOC_HEADER = """
orion:
  Orion cli script for asynchronous distributed optimization

"""


def load_modules_parser(orion_parser):
    """Search through the `cli` folder for any module containing a `get_parser` function"""
    modules = module_import.load_modules_in_path('orion.core.cli',
                                                 lambda m: hasattr(m, 'get_parser'))

    for m in modules:
        get_parser = getattr(m, 'get_parser')
        get_parser(orion_parser.get_subparsers())


def main(argv=None):
    """Entry point for `orion.core` functionality."""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help

    orion_parser = OrionArgsParser(CLI_DOC_HEADER)

    load_modules_parser(orion_parser)

    orion_parser.execute(argv)

    return 0


<<<<<<< HEAD
=======
def load_modules_parser(orion_parser):
    """Search through the `cli` folder for any module containing a `get_parser` function"""
    modules = module_import.load_modules_in_path('orion.core.cli',
                                                 lambda m: hasattr(m, 'get_parser'))

    for module in modules:
        get_parser = getattr(module, 'get_parser')
        get_parser(orion_parser.get_subparsers())


>>>>>>> 2a224b4... Fix some flake8 and pylint issues
if __name__ == "__main__":
    main()
