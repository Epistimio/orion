#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module running the setup command
================================

Creates a configurarion file for the database.

"""

import logging

from orion.core.cli.db.setup import main

log = logging.getLogger(__name__)
DESCRIPTION = "(DEPRECATED) Use command `orion db setup` instead"


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    setup_parser = parser.add_parser("setup", help=DESCRIPTION, description=DESCRIPTION)

    setup_parser.set_defaults(func=main)

    return setup_parser


# pylint: disable = unused-argument
def wrap_main(args):
    """Build a configuration file."""
    log.warning(
        "Command `orion setup` is deprecated and will be removed in v0.2.0. Use "
        "`orion db setup` instead."
    )

    main(args)
