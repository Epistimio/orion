#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to info experiments
==========================

Commandline support to print details of experiments in terminal

"""
import logging
import sys

import orion.core.io.experiment_builder as experiment_builder
from orion.core.cli.base import get_basic_args_group
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
        print("Experiment {} not found in db.".format(args.get("name", None)))
        sys.exit(1)

    print(format_info(experiment))
