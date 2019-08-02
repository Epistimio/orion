#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.status` -- Module to status experiments
============================================================

.. module:: status
   :platform: Unix
   :synopsis: List the trials and their statuses for experiments.

"""
import collections
import logging

import tabulate

from orion.core.cli import base as cli
from orion.core.io.evc_builder import EVCBuilder
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.storage.base import get_storage


log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    status_parser = parser.add_parser('status', help='status help')

    cli.get_basic_args_group(status_parser)

    status_parser.add_argument(
        '-a', '--all', action="store_true",
        help="Show all trials line by line. Otherwise, they are all aggregated by status")

    status_parser.add_argument(
        '-C', '--collapse', action="store_true",
        help=("Aggregate together results of all child experiments. Otherwise they are all "
              "printed hierarchically"))

    status_parser.set_defaults(func=main)

    return status_parser


def main(args):
    """Fetch config and status experiments"""
    builder = ExperimentBuilder()
    local_config = builder.fetch_full_config(args, use_db=False)
    builder.setup_storage(local_config)

    experiments = get_experiments(args)

    if args.get('name'):
        print_status(experiments[0], all_trials=args.get('all'), collapse=args.get('collapse'))
        return

    for exp in filter(lambda e: e.refers.get('parent_id') is None, experiments):
        if args.get('collapse'):
            print_status(exp, all_trials=args.get('all'), collapse=True)
        else:
            print_status_recursively(exp, all_trials=args.get('all'))


def get_experiments(args):
    """Return the different experiments.

    Parameters
    ----------
    args: dict
        Commandline arguments.

    """
    projection = {'name': 1, 'version': 1}

    query = {'name': args['name']} if args.get('name') else {}
    experiments = get_storage().fetch_experiments(query, projection)

    return [EVCBuilder().build_view_from({'name': exp['name'], 'version': exp['version']})
            for exp in experiments]


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
    if collapse:
        trials = exp.fetch_trials_tree({})
    else:
        trials = exp.fetch_trials({})

    exp_title = exp.name + ' - v.{}'.format(exp.version)
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

    headers = ['status', 'quantity']

    lines = []
    for status, c_trials in sorted(status_dict.items()):
        line = [status, len(c_trials)]

        if c_trials[0].objective:
            headers.append('params')
            headers.append('min {}'.format(c_trials[0].objective.name))
            line.append(format_params(trial.params))
            line.append(min(trial.objective.value for trial in c_trials))

        lines.append(line)

    if trials:
        grid = tabulate.tabulate(lines, headers=headers)
        tab = " " * offset
        print(tab + ("\n" + tab).join(grid.split("\n")))
    else:
        print(" " * offset, 'empty', sep="")

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
    headers = ['id', 'status', 'params', 'best objective']
    lines = []

    for trial in sorted(trials, key=lambda t: t.status):
        line = [trial.id, trial.status, format_params(trial.params)]

        if trial.objective:
            headers[-1] = 'min {}'.format(trial.objective.name)
            line.append(trial.objective.value)

        lines.append(line)

    grid = tabulate.tabulate(lines, headers=headers)
    tab = " " * offset
    print(tab + ("\n" + tab).join(grid.split("\n")))

    print("\n")


def format_params(params):
    """Return a formatted line for the parameters of a trial."""
    line = ""

    for param in params:
        param_str = "{}={}".format(param.name[1:], param.value)
        line += (param_str + ", ")

    line = line[:-2]

    return line
