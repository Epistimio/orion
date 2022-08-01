#!/usr/bin/env python
"""
Module running the rm command
=============================

Delete experiments and trials from the database

"""
import argparse
import logging
import sys

from orion.core.io import experiment_builder
from orion.core.utils.pptree import print_tree
from orion.core.utils.terminal import confirm_name

logger = logging.getLogger(__name__)


EXP_RM_MESSAGE = """
All experiments above and their corresponding trials will be deleted.
To select a specific version use --version <VERSION>. Note that all
children of a given version will be deleted. Oríon cannot delete a
parent experiment without deleting the children.
To delete trials only, use --status <STATUS>.

Make sure to stop any worker currently executing one of these experiment.

To proceed, type again the name of the experiment: """


TRIALS_RM_MESSAGE = """
Matching trials of all experiments above will be deleted.
To select a specific version use --version <VERSION>.
Note that trials of all children of a given version will be deleted.

Make sure to stop any worker currently executing one of these experiment.

To proceed, type again the name of the experiment: """


DESCRIPTION = """
Command to delete experiments and trials.

To delete an experiment and its trials, simply give the experiment's name.
$ orion db rm my-exp-name

To delete only trials that are broken, simply add --status broken.
Note that the experiment will not be deleted, only the trials.
$ orion db rm my-exp-name --status broken

Or --status * to delete all trials of the experiment.
$ orion db rm my-exp-name --status *

By default, the last version of the experiment is deleted. Add --version
to select a prior version. Note that all child of the selected version
will be deleted as well. You cannot delete a parent experiment without
deleting the child experiments.
$ orion db rm my-exp-name --version 1
"""


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    rm_parser = parser.add_parser(
        "rm",
        description=DESCRIPTION,
        help="Deletes experiments and trials",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    rm_parser.set_defaults(func=main)

    rm_parser.add_argument("name", help="Name of the experiment to delete.")

    rm_parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="user provided " "orion configuration file",
    )

    rm_parser.add_argument(
        "-v",
        "--version",
        type=int,
        default=None,
        help="specific version of experiment to fetch; "
        "(default: last version matching.)",
    )

    rm_parser.add_argument(
        "-s",
        "--status",
        help="Remove all trials of the experiment with the given status "
        "(Will not delete the experiment). "
        "Also supports --status=* to delete all trials of a given experiment.",
    )

    rm_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force delete without asking to enter experiment name twice.",
    )

    return rm_parser


def process_trial_rm(storage, root, status):
    """Delete the matching trials of the given experiment."""
    trials_total = 0
    for node in root:
        if status == "*":
            query = {}
        else:
            query = {"status": status}

        count = storage.delete_trials(uid=node.item.id, where=query)
        logger.debug(
            "%d trials deleted in experiment %s-v%d",
            count,
            node.item.name,
            node.item.version,
        )
        trials_total += count

    print(f"{trials_total} trials deleted")


def process_exp_rm(storage, root):
    """Delete the given experiment node and all its children."""
    trials_total = 0
    exp_total = 0
    for node in root:
        count = storage.delete_trials(uid=node.item.id)
        trials_total += count
        logger.debug(
            "%d trials deleted in experiment %s-v%d",
            count,
            node.item.name,
            node.item.version,
        )
        count = storage.delete_algorithm_lock(uid=node.item.id)
        logger.debug(
            "%s algorithm lock for experiment %s-v%d deleted",
            count,
            node.item.name,
            node.item.version,
        )
        count = storage.delete_experiment(uid=node.item.id)
        logger.debug(
            "%s experiment %s-v%d deleted", count, node.item.name, node.item.version
        )
        exp_total += count

    print(f"{trials_total} trials deleted")
    print(f"{exp_total} experiments deleted")


def delete_experiments(storage, root, name, force):
    """Delete matching experiments after user confirmation."""
    confirmed = confirm_name(EXP_RM_MESSAGE, name, force)

    if not confirmed:
        print("Confirmation failed, aborting operation.")
        sys.exit(1)

    process_exp_rm(storage, root)


def delete_trials(storage, root, name, status, force):
    """Delete all matching trials after user confirmation."""
    confirmed = confirm_name(TRIALS_RM_MESSAGE, name, force)

    if not confirmed:
        print("Confirmation failed, aborting operation.")
        sys.exit(1)

    process_trial_rm(storage, root, status)


def main(args):
    """Remove the experiment(s) or trial(s)."""
    config = experiment_builder.get_cmd_config(args)
    builder = experiment_builder.ExperimentBuilder(config.get("storage"))

    # Find root experiment
    root = builder.load(name=args["name"], version=args.get("version", None)).node

    # List all experiments with children
    print_tree(root, nameattr="tree_name")

    storage = builder.storage

    if args["status"]:
        delete_trials(storage, root, args["name"], args["status"], args["force"])
    else:
        delete_experiments(storage, root, args["name"], args["force"])
