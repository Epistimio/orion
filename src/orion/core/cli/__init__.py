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
from orion.core.utils import module_import

log = logging.getLogger(__name__)


def load_modules_parser(orion_parser):
    """Search through the `cli` folder for any module containing a `get_parser` function"""
    modules = module_import.load_modules_in_path('orion.core.cli',
                                                 lambda m: hasattr(m, 'add_subparser'))
    # modules = module_import.load_modules_in_path('orion.core.cli')

    for module in modules:
        get_parser = getattr(module, 'add_subparser')
        get_parser(orion_parser.get_subparsers())


def main(argv=None):
    """Entry point for `orion.core` functionality."""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help

    orion_parser = OrionArgsParser()

    load_modules_parser(orion_parser)

    argv = ["-v", "hunt", "--config", "/Users/xuechao/aaa/work/gitRepo/orion_ibm/local_mongo_config.yaml",
            "-n", "test61", "--max-trials", "100",
            "/Users/xuechao/aaa/work/gitRepo/orion_ibm/dummy_orion/dummy_orion.py",
            "--x0~uniform(0, 1, discrete=False)", "--x1~uniform(0, 2, discrete=True)",
            "--x2~uniform(0, 1, discrete=False)", "--x3~choices([0, 3, 1])",
            "--x4~fidelity(1, 16, 2)"]

    return orion_parser.execute(argv)


if __name__ == "__main__":
    main()
