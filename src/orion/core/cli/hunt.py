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

import orion.core
from orion.core.cli import base as cli
from orion.core.cli import evc as evc_cli
import orion.core.io.experiment_builder as experiment_builder
from orion.core.worker import workon

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    hunt_parser = parser.add_parser('hunt', help='hunt help')

    orion_group = cli.get_basic_args_group(
        hunt_parser, group_name='Hunt arguments', group_help='')

    orion.core.config.experiment.add_arguments(
        orion_group,
        rename=dict(max_broken='--exp-max-broken', max_trials='--exp-max-trials'))

    orion_group.add_argument(
        '--max-trials', type=int, metavar='#',
        help="(DEPRECATED) This argument will be removed in v0.3. Use --exp-max-trials instead")

    worker_args_group = hunt_parser.add_argument_group(
        "Worker arguments (optional)",
        description="Arguments to automatically resolved branching events.")

    orion.core.config.worker.add_arguments(
        worker_args_group,
        rename=dict(max_broken='--worker-max-broken', max_trials='--worker-max-trials'))

    evc_cli.get_branching_args_group(hunt_parser)

    cli.get_user_args_group(hunt_parser)

    hunt_parser.set_defaults(func=main)
    hunt_parser.set_defaults(help_empty=True)  # Print help if command is empty

    return hunt_parser


def main(args):
    """Build experiment and execute hunt command"""
    args['root'] = None
    args['leafs'] = []
    # TODO: simplify when parameter parsing is refactored
    experiment = experiment_builder.build_from_args(args)
    config = experiment_builder.get_cmd_config(args)
    worker_config = orion.core.config.worker.to_dict()
    if config.get('worker'):
        worker_config.update(config.get('worker'))

    workon(experiment, **worker_config)
