#!/usr/bin/env python
"""
Module running the init_only command
====================================

Creates a new experiment.

"""

import logging

import orion.core
from orion.core.cli import base as cli
from orion.core.cli import evc as evc_cli
from orion.core.io import experiment_builder

log = logging.getLogger(__name__)
DESCRIPTION = "(DEPRECATED) Use command `orion hunt --init_only` instead"


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    init_only_parser = parser.add_parser(
        "init_only", help=DESCRIPTION, description=DESCRIPTION
    )

    orion_group = cli.get_basic_args_group(
        init_only_parser, group_name="init_only arguments", group_help=""
    )

    orion.core.config.experiment.add_arguments(
        orion_group,
        rename=dict(max_broken="--exp-max-broken", max_trials="--exp-max-trials"),
    )

    orion_group.add_argument(
        "--max-trials",
        type=int,
        metavar="#",
        help="(DEPRECATED) This argument will be removed in v0.3. Use --exp-max-trials instead",
    )

    evc_cli.get_branching_args_group(init_only_parser)

    cli.get_user_args_group(init_only_parser)

    init_only_parser.set_defaults(func=main)
    init_only_parser.set_defaults(help_empty=True)  # Print help if command is empty

    return init_only_parser


def main(args):
    """Build and initialize experiment"""
    # By building the experiment, we create a new experiment document in database
    log.warning(
        "Command init_only is deprecated and will be removed in v0.3. "
        "Use orion hunt --init-only instead."
    )
    experiment_builder.build_from_args(args)
