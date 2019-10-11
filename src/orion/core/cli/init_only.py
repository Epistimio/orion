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

from orion.core.cli import base as cli
from orion.core.cli import evc as evc_cli
from orion.core.io.experiment_builder import ExperimentBuilder, get_storage

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    init_only_parser = parser.add_parser('init_only', help='init_only help')

    cli.get_basic_args_group(init_only_parser)

    evc_cli.get_branching_args_group(init_only_parser)

    cli.get_user_args_group(init_only_parser)

    init_only_parser.set_defaults(func=main)

    return init_only_parser


def main(args):
    """Build and initialize experiment"""
    # By building the experiment, we create a new experiment document in database
    ExperimentBuilder().build_from(args)
