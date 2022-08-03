#!/usr/bin/env python
"""
Module running the release command
==================================

Release the lock of a given experiment.

"""
import argparse
import logging
import sys

from orion.core.io import experiment_builder
from orion.core.utils.pptree import print_tree
from orion.core.utils.terminal import confirm_name

logger = logging.getLogger(__name__)


DESCRIPTION = """
Command to force the release of the algorithm lock of an experiment.
"""


CONFIRM_MESSAGE = """
Algorithm lock of experiment {experiment.name}-{experiment.version} above will be released.
To select a specific version use --version <VERSION>.

Make sure to stop any worker currently executing one of these experiment.

To proceed, type again the name of the experiment: """


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    set_parser = parser.add_parser(
        "release",
        description=DESCRIPTION,
        help="Release the algorithm lock of an experiment.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    set_parser.set_defaults(func=main)

    set_parser.add_argument(
        "name", help="Name of the experiment to release algorithm lock."
    )

    set_parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="user provided orion configuration file",
    )

    set_parser.add_argument(
        "-v",
        "--version",
        type=int,
        default=None,
        help="specific version of experiment to fetch; "
        "(default: last version matching.)",
    )

    set_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force modify without asking to enter experiment name twice.",
    )

    return set_parser


def process_release_lock(storage, root):
    """Release the lock of the given experiment node and all its children."""
    count = storage.release_algorithm_lock(uid=root.item.id)
    if count:
        print("Algorithm lock successfully released.")
    else:
        print(
            "Release of algorithm lock failed. Make sure the experiment is not being "
            "executed when attempting to release the lock. "
        )


def release_locks(storage, root, name, force):
    """Release locks of matching experiments after user confirmation."""
    confirmed = confirm_name(CONFIRM_MESSAGE.format(experiment=root), name, force)

    if not confirmed:
        print("Confirmation failed, aborting operation.")
        sys.exit(1)

    process_release_lock(storage, root)


def main(args):
    """Remove the experiment(s) or trial(s)."""
    config = experiment_builder.get_cmd_config(args)
    builder = experiment_builder.ExperimentBuilder(config.get("storage"))

    # Find root experiment
    root = builder.load(name=args["name"], version=args.get("version", None)).node

    # List all experiments with children
    print_tree(root, nameattr="tree_name")

    storage = builder.storage

    release_locks(storage, root, args["name"], args["force"])
