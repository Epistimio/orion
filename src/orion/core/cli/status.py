#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module to status experiments
============================

List the trials and their statuses for experiments.

"""
import collections
import logging

import tabulate

import orion.core.io.experiment_builder as experiment_builder
from orion.core.cli import base as cli
from orion.storage.base import get_storage

log = logging.getLogger(__name__)
SHORT_DESCRIPTION = "Gives an overview of experiments' trials"
DESCRIPTION = """
This command outputs the status of the different trials inside every experiment or a
specific EVC tree. It can either give you an overview of the different trials status, i.e.,
the number of currently completed trials and so on, or, it can give you a deeper view of the
experiments by outlining every single trial, its status and its objective.
"""


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    status_parser = parser.add_parser(
        "status", help=SHORT_DESCRIPTION, description=DESCRIPTION
    )

    cli.get_basic_args_group(status_parser)

    status_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Show all trials line by line. Otherwise, they are all aggregated by status",
    )

    status_parser.add_argument(
        "-C",
        "--collapse",
        action="store_true",
        help=(
            "Aggregate together results of all child experiments. Otherwise they are all "
            "printed hierarchically"
        ),
    )

    status_parser.add_argument(
        "-e",
        "--expand-versions",
        action="store_true",
        help=(
            "Show all the version of every experiments instead of only the latest one"
        ),
    )

    status_parser.set_defaults(func=main)

    return status_parser


def main(args):
    """Fetch config and status experiments"""
    config = experiment_builder.get_cmd_config(args)
    experiment_builder.setup_storage(config.get("storage"))

    args["all_trials"] = args.pop("all", False)

    experiments = get_experiments(args)

    if not experiments:
        print("No experiment found")
        return

    if args.get("name"):
        print_evc([experiments[0]], **args)
        return

    if args.get("version"):
        if args.get("collapse") or args.get("expand_versions"):
            raise RuntimeError(
                "Cannot fetch specific version of experiments with --collapse "
                "or --expand-versions."
            )

    print_evc(experiments, **args)


# pylint: disable=unused-argument
def print_evc(
    experiments,
    version=None,
    all_trials=False,
    collapse=False,
    expand_versions=False,
    **kwargs
):
    """Print each EVC tree

    Parameters
    ----------
    args: dict
        Commandline arguments.

    """
    for exp in experiments:
        experiment = experiment_builder.load(exp.name, version)
        if version is None:
            expand_experiment = exp
        else:
            expand_experiment = experiment
        expand = expand_versions or _has_named_children(expand_experiment)
        if expand and not collapse:
            print_status_recursively(expand_experiment, all_trials=all_trials)
        else:
            print_status(experiment, all_trials=all_trials, collapse=True)


def get_experiments(args):
    """Return the different experiments.

    Parameters
    ----------
    args: dict
        Commandline arguments.

    """
    projection = {"name": 1, "version": 1, "refers": 1}

    query = {"name": args["name"]} if args.get("name") else {}
    experiments = get_storage().fetch_experiments(query, projection)

    if args["name"]:
        root_experiments = experiments
    else:
        root_experiments = [
            exp
            for exp in experiments
            if exp["refers"].get("root_id", exp["_id"]) == exp["_id"]
        ]

    return [
        experiment_builder.load(name=exp["name"], version=exp.get("version", 1))
        for exp in root_experiments
    ]


def _has_named_children(exp):
    return any(node.name != exp.name for node in exp.node)


def print_status_recursively(exp, depth=0, **kwargs):
    """Print the status recursively of the children of the current experiment.

    Parameters
    ----------
    exp: `orion.core.worker.Experiment`
        The current experiment to print.
    depth: int
        The current depth of the tree.

    """
    print_status(exp, offset=depth * 2, **kwargs)

    for child in exp.node.children:
        print_status_recursively(child.item, depth + 1, **kwargs)


def print_status(exp, offset=0, all_trials=False, collapse=False):
    """Print the status of the current experiment.

    Parameters
    ----------
    offset: int, optional
        The number of tabs to the right this experiment is.
    all_trials: bool, optional
        Print all trials individually
    collapse: bool, optional
        Fetch trials for entire EVCTree. Defaults to False.

    """
    trials = exp.fetch_trials(with_evc_tree=collapse)

    exp_title = exp.node.tree_name
    print(" " * offset, exp_title, sep="")
    print(" " * offset, "=" * len(exp_title), sep="")

    if all_trials:
        print_all_trials(trials, offset=offset)
    else:
        print_summary(trials, offset=offset)


def print_summary(trials, offset=0):
    """Print a summary of the current experiment.

    Parameters
    ----------
    trials: list
        Trials to summarize.
    offset: int, optional
        The number of tabs to the right this experiment is.

    """
    status_dict = collections.defaultdict(list)

    for trial in trials:
        status_dict[trial.status].append(trial)

    headers = ["status", "quantity"]

    lines = []
    for status, c_trials in sorted(status_dict.items()):
        line = [status, len(c_trials)]

        if c_trials[0].objective:
            headers.append("min {}".format(c_trials[0].objective.name))
            line.append(
                min(trial.objective.value for trial in c_trials if trial.objective)
            )

        lines.append(line)

    if trials:
        grid = tabulate.tabulate(lines, headers=headers)
        tab = " " * offset
        print(tab + ("\n" + tab).join(grid.split("\n")))
    else:
        print(" " * offset, "empty", sep="")

    print("\n")


def print_all_trials(trials, offset=0):
    """Print all trials of the current experiment individually.

    Parameters
    ----------
    trials: list
        Trials to list in terminal.
    offset: int, optional
        The number of tabs to the right this experiment is.

    """
    headers = ["id", "status", "best objective"]
    lines = []

    for trial in sorted(trials, key=lambda t: t.status):
        line = [trial.id, trial.status]

        if trial.objective:
            headers[-1] = "min {}".format(trial.objective.name)
            line.append(trial.objective.value)

        lines.append(line)

    if not trials:
        lines.append(["empty", "", ""])

    grid = tabulate.tabulate(lines, headers=headers)
    tab = " " * offset
    print(tab + ("\n" + tab).join(grid.split("\n")))

    print("\n")
