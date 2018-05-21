#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=eval-used,protected-access
"""
:mod:`orion.core.cli.viz` -- Module to insert new trials
===========================================================

.. module:: insert
   :platform: Unix
   :synopsis: Insert creates new trials for a given experiment with fixed values

"""
import logging

import orion
from orion.core.cli import resolve_config
from orion.viz.evc.graph import graph_vizualize
from orion.viz.evc.text import text_vizualize

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    viz_parser = parser.add_parser('viz', help='Vizualization help')

    resolve_config.get_basic_args_group(viz_parser)

    resolve_config.get_user_args_group(viz_parser)

    viz_parser.set_defaults(func=main)

    return viz_parser


def main(args):
    """Fetch config and initialize experiment"""
    # Note: Side effects on args
    config = fetch_config(args)

    _execute(args, config)


def fetch_config(args):
    """Create the dictionary of modified args for the execution of the command"""
    # Explicitly add orion's version as experiment's metadata
    args['metadata'] = dict()
    args['metadata']['orion_version'] = orion.core.__version__
    log.debug("Using orion version %s", args['metadata']['orion_version'])

    config = resolve_config.fetch_config(args)

    args.pop('user_script')
    args['metadata']['user_args'] = args.pop('user_args')

    return config


# By inferring the experiment, we create a new configured experiment
def _execute(cmdargs, cmdconfig):
    graph_vizualize(cmdargs, cmdconfig)
    text_vizualize(cmdargs, cmdconfig)

def graph_vizualize():
    pass

def text_vizualize():
    pass
