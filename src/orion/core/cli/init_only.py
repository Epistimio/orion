#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.init_only` -- Module running the init_only command
=======================================================================

.. module:: init_only
   :platform: Unix
   :synopsis: Creates a new experiment.
"""

import logging

import orion
from orion.core.cli import base as cli
from orion.core.io.experiment_builder import ExperimentBuilder

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    init_only_parser = parser.add_parser('init_only', help='init_only help')

    cli.get_basic_args_group(init_only_parser)

    cli.get_user_args_group(init_only_parser)

    init_only_parser.set_defaults(func=main)

    return init_only_parser


def main(args):
    """Set metadata and initialize experiment"""
    set_metadata(args)

    _execute(args)


def set_metadata(args):
    """Set metadata from command line arguments."""
    # Explicitly add orion's version as experiment's metadata
    args['metadata'] = dict()
    args['metadata']['orion_version'] = orion.core.__version__
    log.debug("Using orion version %s", args['metadata']['orion_version'])

    args.pop('user_script')
    args['metadata']['user_args'] = args.pop('user_args')


# By building the experiment, we create a new experiment document in database
def _execute(cmdargs):
    ExperimentBuilder().build_from(cmdargs)
