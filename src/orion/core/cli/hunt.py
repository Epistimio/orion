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

from orion.core.cli import base as cli
from orion.core.io import resolve_config
from orion.core.io.evc_builder import EVCBuilder
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

    orion_group.add_argument(
        '--branch', '-b', type=str, metavar='newBranchID',
        help="name of the new experiment resulting of the branch of this one.")

    cli.get_user_args_group(hunt_parser)

    hunt_parser.set_defaults(func=main)

    return hunt_parser


def main(args):
    """Fetch config and execute hunt command"""
    args['root'] = None
    args['leafs'] = []
    experiment = EVCBuilder().build_from(args)
    workon(experiment)
