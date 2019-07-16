#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.status` -- Module to status experiments
============================================================

.. module:: status
   :platform: Unix
   :synopsis: List experiments in terminal

"""
import collections
import logging

import tabulate

from orion.core.cli import base as cli
from orion.core.io.database import Database
from orion.core.io.evc_builder import EVCBuilder
from orion.core.io.experiment_builder import ExperimentBuilder

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    status_parser = parser.add_parser('status', help='status help')

    cli.get_basic_args_group(status_parser)

    status_parser.add_argument(
        '-a', '--all', action="store_true",
        help="Show all trials line by line. Otherwise, they are all aggregated by status")

    status_parser.add_argument(
        '-r', '--recursive', action="store_true",
        help="Divide trials per experiments hierarchically. Otherwise they are all aggregated in "
             "parent experiment")

    status_parser.set_defaults(func=main)

    return status_parser


def main(args):
    """Fetch config and status experiments"""
    builder = ExperimentBuilder()
    local_config = builder.fetch_full_config(args, use_db=False)
    builder.setup_database(local_config)

    experiments = get_experiments(args)

    if args.get('recursive'):
        for exp in filter(lambda e: e.refers['parent_id'] is None, experiments):
            print_status_recursively(exp, all_trials=args.get('all'))
    else:
        for exp in experiments:
            print_status(exp, all_trials=args.get('all'))


def get_experiments(args):
    """Return the different experiments.

    Parameters
    ----------
    args: dict
        Commandline arguments.

    """
    projection = {'name': 1}

    query = {'name': args['name']} if args.get('name') else {}
    experiments = Database().read("experiments", query, projection)

    return [EVCBuilder().build_from({'name': exp['name']}) for exp in experiments]


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


def print_status(exp, offset=0, all_trials=False):
    """Print the status of the current experiment.

    Parameters
    ----------
    offset: int, optional
        The number of tabs to the right this experiment is.
    all_trials: bool, optional
        Print all trials individually

    """
    if all_trials:
        print_all_trials(exp, offset=offset)
    else:
        print_summary(exp, offset=offset)


def print_summary(exp, offset=0):
    """Print a summary of the current experiment.

    Parameters
    ----------
    offset: int, optional
        The number of tabs to the right this experiment is.

    """
    status_dict = collections.defaultdict(list)
    name = exp.name
    trials = exp.fetch_trials({})

    for trial in trials:
        status_dict[trial.status].append(trial)

    print(" " * offset, name, sep="")
    print(" " * offset, "=" * len(name), sep="")

    headers = ['status', 'quantity']

    lines = []
    for status, trials in sorted(status_dict.items()):
        line = [status, len(trials)]

        if trials[0].objective:
            headers.append('min {}'.format(trials[0].objective.name))
            line.append(min(trial.objective.value for trial in trials))

        lines.append(line)

    if trials:
        grid = tabulate.tabulate(lines, headers=headers)
        tab = " " * offset
        print(tab + ("\n" + tab).join(grid.split("\n")))
    else:
        print(" " * offset, 'empty', sep="")

    print("\n")


def print_all_trials(exp, offset=0):
    """Print all trials of the current experiment individually.

    Parameters
    ----------
    offset: int, optional
        The number of tabs to the right this experiment is.

    """
    name = exp.name
    trials = exp.fetch_trials({})

    print(" " * offset, name, sep="")
    print(" " * offset, "=" * len(name), sep="")
    headers = ['id', 'status', 'best objective']
    lines = []

    for trial in sorted(trials, key=lambda t: t.status):
        line = [trial.id, trial.status]

        if trial.objective:
            headers[-1] = 'min {}'.format(trial.objective.name)
            line.append(trial.objective.value)

        lines.append(line)

    grid = tabulate.tabulate(lines, headers=headers)
    tab = " " * offset
    print(tab + ("\n" + tab).join(grid.split("\n")))

    print("\n")
