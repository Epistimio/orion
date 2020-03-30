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

# from cli.base import OrionArgsParser
# from ..utils import module_import

log = logging.getLogger(__name__)


def load_modules_parser(orion_parser):
    """Search through the `cli` folder for any module containing a `get_parser` function"""
    modules = module_import.load_modules_in_path('orion.core.cli', lambda m: hasattr(m, 'add_subparser'))
    # modules = module_import.load_modules_in_path('orion.core.cli')

    for module in modules:
        get_parser = getattr(module, 'add_subparser')
        get_parser(orion_parser.get_subparsers())


def main(argv=None):
    """Entry point for `orion.core` functionality."""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help

    orion_parser = OrionArgsParser()

    # orion_parser.parser.add_argument(
    #     '-v', '--verbose',
    #      default=0,)

    load_modules_parser(orion_parser)

    # argv="-v hunt -n orion-tutorial --max-trials 5 ./main.py --lr~'loguniform(1e-5, 1.0)'"

    # argv = ["-v", "hunt", "-n", "orion-tutorial2", "--max-trials", "2",
    #         "/home/xuechao/orion_ibm/examples/mnist/main.py", "--lr~loguniform(1e-5, 1.0)", "--gamma~uniform(0.2, 0.7)"]

    argv = ["-v", "hunt", "-n", "orion-tutorial9", "--max-trials", "2",
            "/home/xuechao/orion_ibm/examples/mnist/main.py",
            "--lr~loguniform(5e-3, 5e-2)", "--gamma~uniform(0.2, 0.7)"]

    return orion_parser.execute(argv)


if __name__ == "__main__":
    main()
