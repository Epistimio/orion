#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.hunt` -- Module running the optimization command
=====================================================================

.. module:: hunt
   :platform: Unix
   :synopsis: Gets an experiment and iterates over it until one of the exit conditions is met

"""

import logging
import os

import orion
from orion.core.cli import base as cli
from orion.core.io import resolve_config
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.worker import workon

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    hunt_parser = parser.add_parser('hunt', help='hunt help')

    orion_group = cli.get_basic_args_group(hunt_parser)

    orion_group.add_argument(
        '--max-trials', type=int, metavar='#',
        help="number of jobs/trials to be completed "
             "(default: %s)" % resolve_config.DEF_CMD_MAX_TRIALS[1])

    orion_group.add_argument(
        "--pool-size", type=int, metavar='#',
        help="number of concurrent workers to evaluate candidate samples "
             "(default: %s)" % resolve_config.DEF_CMD_POOL_SIZE[1])

    cli.get_user_args_group(hunt_parser)

    hunt_parser.set_defaults(func=main)

    return hunt_parser


def main(args):
    """Fetch config and execute hunt command"""
    # Note: Side effects on args
    set_metadata(args)

    _execute(args)


def set_metadata(args):
    """Set metadata from command line arguments."""
    # Explicitly add orion's version as experiment's metadata
    args['metadata'] = dict()
    args['metadata']['orion_version'] = orion.core.__version__
    log.debug("Using orion version %s", args['metadata']['orion_version'])

    # Move 'user_script' and 'user_args' to 'metadata' key
    user_script = args.pop('user_script')
    abs_user_script = os.path.abspath(user_script)
    if resolve_config.is_exe(abs_user_script):
        user_script = abs_user_script

    args['metadata']['user_script'] = user_script
    args['metadata']['user_args'] = args.pop('user_args')
    log.debug("Problem definition: %s %s", args['metadata']['user_script'],
              ' '.join(args['metadata']['user_args']))


def _execute(cmdargs):
    experiment = ExperimentBuilder().build_from(cmdargs)
    workon(experiment)
