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
import os

from orion.core.cli import resolve_config

log = logging.getLogger(__name__)

CLI_DOC_HEADER = """
orion:
  Orion cli script for asynchronous distributed optimization

"""


def main(argv=None):
    """Entry point for `orion.core` functionality."""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help

    orion_parser = resolve_config.OrionArgsParser(CLI_DOC_HEADER)

    load_modules_parser(orion_parser)

    orion_parser.execute(argv)

    return 0


def load_modules_parser(orion_parser):
    """Search through the `cli` folder for any module containing a `add_subparser` function"""
    this_module = __import__('orion.core.cli', fromlist=[''])
    path = this_module.__path__[0]

    files = list(map(lambda f: f.split('.')[0],
                     filter(lambda f2: f2.endswith('py'), os.listdir(path))))

    for f in files:
        module = __import__('orion.core.cli.' + f, fromlist=[''])

        if hasattr(module, 'add_subparser'):
            add_subparser = getattr(module, 'add_subparser')
            add_subparser(orion_parser.get_subparsers())


if __name__ == "__main__":
    main()
