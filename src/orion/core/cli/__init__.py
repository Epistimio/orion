#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions that define console scripts
=====================================

Helper functions to setup an experiment and execute it.

"""
import logging

from orion.core.cli.base import OrionArgsParser
from orion.core.utils import module_import

log = logging.getLogger(__name__)


def load_modules_parser(orion_parser):
    """Search through the `cli` folder for any module containing a `get_parser` function"""
    modules = module_import.load_modules_in_path(
        "orion.core.cli", lambda m: hasattr(m, "add_subparser")
    )
    for module in modules:
        get_parser = getattr(module, "add_subparser")
        get_parser(orion_parser.get_subparsers())


def main(argv=None):
    """Entry point for `orion.core` functionality."""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help

    orion_parser = OrionArgsParser()

    load_modules_parser(orion_parser)

    return orion_parser.execute(argv)


if __name__ == "__main__":
    returncode = main()
    if returncode > 0:
        raise SystemExit(returncode)
