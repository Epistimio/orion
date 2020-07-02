#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.info` -- Module to info experiments
========================================================

.. module:: info
   :platform: Unix
   :synopsis: Commandline support to print details of experiments in terminal

"""
import logging
import sys

from orion.core.cli.base import get_basic_args_group
import orion.core.io.experiment_builder as experiment_builder
from orion.core.utils.format_terminal import format_info

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    info_parser = parser.add_parser('info', help='info help')
    get_basic_args_group(info_parser)

    info_parser.set_defaults(func=main)

    return info_parser


def main(args):
    """Fetch config and info experiments"""
    try:
        experiment = experiment_builder.build_view_from_args(args)
    except ValueError:
        print('Experiment {} not found in db.'.format(args.get('name', None)))
        sys.exit(1)

    print(format_info(experiment))
