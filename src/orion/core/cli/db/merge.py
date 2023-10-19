#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module running the set command
==============================

Update data of experiments and trials in the storage

"""
import argparse
import logging

import orion.core.io.experiment_builder as experiment_builder

logger = logging.getLogger(__name__)


DESCRIPTION = """
"""


CONFIRM_MESSAGE = """
"""


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    set_parser = parser.add_parser(
        "merge",
        description=DESCRIPTION,
        help="",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    set_parser.set_defaults(func=main)

    set_parser.add_argument("base", help="Name of the base experiment.")

    set_parser.add_argument("experiment", help="Name of the experiment to merge.")

    set_parser.add_argument("merged", help="Name of the merged experiment.")

    set_parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="user provided orion configuration file",
    )
    set_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force merge without asking to enter experiment name twice.",
    )

    return set_parser


def main(args):
    """Remove the experiment(s) or trial(s)."""
    config = experiment_builder.get_cmd_config(args)
    experiment_builder.setup_storage(config.get("storage"))

    # Load base experiment
    base = experiment_builder.load(
        name=args["base"]
    )

    # Load experiment to merge
    exp = experiment_builder.load(
        name=args["experiment"]
    )

    # Make merged experiment
    merged_config = base.configuration
    del merged_config["_id"]
    del merged_config["refers"]
    merged_config["name"] = args["merged"]
    merged = experiment_builder.create_experiment(mode="w", **merged_config)

    base_trials = base.fetch_trials(with_evc_tree=True)
    exp_trials = exp.fetch_trials(with_evc_tree=True)

    for trial in base.fetch_trials(with_evc_tree=True):
        trial.id_override = None
        merged.register_trial(trial, status=trial.status)

    for trial in exp.fetch_trials(with_evc_tree=True):
        trial.id_override = None
        merged.register_trial(trial, status=trial.status)

    return 0
