#!/usr/bin/env python
"""
Module to info experiments
==========================

Commandline support to print details of experiments in terminal

"""
import logging
import sys

from orion.core.cli.base import get_basic_args_group
from orion.core.io import experiment_builder
from orion.core.utils.format_terminal import format_info

log = logging.getLogger(__name__)
SHORT_DESCRIPTION = "Gives detailed information about experiments"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    info_parser = parser.add_parser(
        "info", help=SHORT_DESCRIPTION, description=SHORT_DESCRIPTION
    )
    get_basic_args_group(info_parser)

    info_parser.set_defaults(func=main)

    return info_parser


def main(args):
    """Fetch config and info experiments"""
    try:
        experiment = experiment_builder.get_from_args(args, mode="r")
    except ValueError:
        print(f"Experiment {args.get('name', None)} not found in db.")
        sys.exit(1)

    print(format_info(experiment))
