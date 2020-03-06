#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.list` -- Module to list experiments
========================================================
.. module:: list
   :platform: Unix
   :synopsis: List experiments in terminal
"""
import logging

from orion.core.cli import base as cli
import orion.core.io.experiment_builder as experiment_builder
from orion.core.utils.pptree import print_tree
from orion.storage.base import get_storage

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    list_parser = parser.add_parser('list', help='list help')

    cli.get_basic_args_group(list_parser)

    list_parser.set_defaults(func=main)

    return list_parser


def main(args):
    """List all experiments inside database."""
    config = experiment_builder.get_cmd_config(args)
    experiment_builder.setup_storage(config.get('storage'))

    query = {}

    if args['name']:
        query['name'] = args['name']

    experiments = get_storage().fetch_experiments(query)

    if args['name']:
        root_experiments = experiments
    else:
        root_experiments = [exp for exp in experiments
                            if exp['refers'].get('root_id', exp['_id']) == exp['_id']]

    if not root_experiments:
        print("No experiment found")
        return

    for root_experiment in root_experiments:
        root = experiment_builder.build_view(name=root_experiment['name'],
                                             version=root_experiment.get('version')).node
        print_tree(root, nameattr='tree_name')
