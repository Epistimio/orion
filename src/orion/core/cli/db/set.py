#!/usr/bin/env python
"""
Module running the set command
==============================

Update data of experiments and trials in the storage

"""
import argparse
import logging
import sys

# pylint:disable=consider-using-from-import
import orion.core.io.experiment_builder as experiment_builder
from orion.core.utils.pptree import print_tree
from orion.core.utils.terminal import confirm_name

logger = logging.getLogger(__name__)


DESCRIPTION = """
Command to update trial attributes.

To change a trial status, simply give the experiment name,
trial id and status. (use `orion status --all` to get trial ids)
$ orion db set my-exp-name id=3cc91e851e13281ca2152c19d888e937 status=interrupted

To change all trials from a given status to another, simply give the two status
$ orion db set my-exp-name status=broken status=interrupted

Or `*` to apply the change to all trials
$ orion db set my-exp-name '*' status=interrupted

By default, trials of the last version of the experiment are selected.
Add --version to select a prior version. Note that the modification
is applied recursively to all child experiment, but not to the parents.
$ orion db set my-exp-name --version 1 status=broken status=interrupted
"""


CONFIRM_MESSAGE = """
Trials matching the query `{query}` for all experiments listed above
will be modified with `{update}`.
To select a specific version use --version <VERSION>.

Make sure to stop any worker currently executing one of these experiment.

To proceed, type again the name of the experiment: """


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    set_parser = parser.add_parser(
        "set",
        description=DESCRIPTION,
        help="Update trials' attributes",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    set_parser.set_defaults(func=main)

    set_parser.add_argument("name", help="Name of the experiment to delete.")

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

    set_parser.add_argument(
        "query",
        help=(
            f"Query for trials to update. Can be `*` to update all trials. "
            f"Otherwise format must be <attribute>=<value>. Ex: status=broken. "
            f"Supported attributes are {VALID_QUERY_ATTRS}"
        ),
    )

    set_parser.add_argument(
        "update",
        help=(
            f"Update for trials. Format must be <attribute>=<value>. "
            f"Ex: status=interrupted. "
            f"Supported attributes are {VALID_UPDATE_ATTRS}"
        ),
    )

    return set_parser


def process_updates(storage, root, query, update):
    """Update the matching trials of the given experiment and its children."""
    trials_total = 0
    for node in root:
        count = storage.update_trials(node.item, where=query, **update)
        logger.debug(
            "%d trials modified in experiment %s-v%d",
            count,
            node.item.name,
            node.item.version,
        )
        trials_total += count

    print(f"{trials_total} trials modified")


VALID_QUERY_ATTRS = ["status", "id"]
VALID_UPDATE_ATTRS = ["status"]


def build_query(experiment, query):
    """Convert query string to dict format

    String format must be <attr name>=<value>
    """
    if query.strip() == "*":
        return {}

    attribute, value = query.split("=")

    if attribute not in VALID_QUERY_ATTRS:
        raise ValueError(
            f"Invalid query attribute `{attribute}`. Must be one of {VALID_QUERY_ATTRS}"
        )

    query = {attribute: value}

    if attribute == "id":
        query["experiment"] = experiment.id

    return query


def build_update(update):
    """Convert update string to dict format

    String format must be <attr name>=<value>
    """
    attribute, value = update.split("=")

    if attribute not in VALID_UPDATE_ATTRS:
        raise ValueError(
            f"Invalid update attribute `{attribute}`. Must be one of {VALID_UPDATE_ATTRS}"
        )

    return {attribute: value}


def main(args):
    """Remove the experiment(s) or trial(s)."""
    config = experiment_builder.get_cmd_config(args)
    builder = experiment_builder.ExperimentBuilder(config.get("storage"))

    # Find root experiment
    root = builder.load(name=args["name"], version=args.get("version", None)).node

    try:
        query = build_query(root.item, args["query"])
        update = build_update(args["update"])
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # List all experiments with children
    print_tree(root, nameattr="tree_name")

    confirmed = confirm_name(
        CONFIRM_MESSAGE.format(query=args["query"], update=args["update"]),
        args["name"],
        args["force"],
    )

    if not confirmed:
        print("Confirmation failed, aborting operation.")
        return 1

    process_updates(builder.storage, root, query, update)

    return 0
