#!/usr/bin/env python
"""
Module to list experiments
==========================

List experiments in terminal

"""
import logging

from orion.core.cli import base as cli
from orion.core.io import experiment_builder
from orion.core.utils.pptree import print_tree

log = logging.getLogger(__name__)
SHORT_DESCRIPTION = "Gives a list of experiments"
DESCRIPTION = """
This command gives a list of your experiments in a easy-to-read tree-like structure.
"""


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    list_parser = parser.add_parser(
        "list", help=SHORT_DESCRIPTION, description=DESCRIPTION
    )

    cli.get_basic_args_group(list_parser)

    list_parser.set_defaults(func=main)

    return list_parser


def main(args):
    """List all experiments inside database."""
    config = experiment_builder.get_cmd_config(args)
    builder = experiment_builder.ExperimentBuilder(config.get("storage"))

    query = {}

    if args["name"]:
        query["name"] = args["name"]
        query["version"] = args.get("version", None) or 1

    experiments = builder.storage.fetch_experiments(query)

    if args["name"]:
        root_experiments = experiments
    else:
        root_experiments = [
            exp
            for exp in experiments
            if exp["refers"].get("root_id", exp["_id"]) == exp["_id"]
        ]

    if not root_experiments:
        print("No experiment found")
        return

    for root_experiment in root_experiments:
        root = builder.load(
            name=root_experiment["name"], version=root_experiment.get("version")
        ).node
        print_tree(root, nameattr="tree_name")
