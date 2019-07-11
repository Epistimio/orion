#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.status` -- Module to status experiments
============================================================

.. module:: status
   :platform: Unix
   :synopsis: List experiments in termnial

"""
import collections
import logging

import tabulate

from orion.core.io.database import Database
from orion.core.cli import base as cli
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.worker.trial import Trial

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

    status_parser.add_argument(
        '-l', '--local', action="store_true",
        help="Show only trials which are running or were running on current machine")

    status_parser.set_defaults(func=main)

    return status_parser


def main(args):
    """Fetch config and status experiments"""
    builder = ExperimentBuilder()
    local_config = builder.fetch_full_config(args, use_db=False)
    builder.setup_database(local_config)

    projection = {'name': 1, 'refers': 1}

    experiments = Database().read("experiments", {}, projection)

    # trials = _build_trials()

    # if args.get('name'):
    #     root_experiments = [e for e in experiments if e['name'] == args['name']]
    # else:
    #     root_experiments = [e for e in experiments
    #                         if e.get('refers') and e['refers']['root_id'] == e['_id']]

    # trees = []
    # for root_experiment in root_experiments:
    #     trees.append(build_experiment_tree(root_experiment, experiments, trials))

    trees = [builder.build_from({'name': exp['name']}) for exp in experiments]
    if args.get('recursive'):
        for tree in trees:
            print_status_recursively(tree, all_trials=args.get('all'))
    else:
        for tree in trees:
            # name, (subtree, trials) = next(iter(tree.items()))
            name = tree.name
            trials = tree.fetch_trials({})
            print_status(name, trials, recursive=True, all_trials=args.get('all'))


def _build_trials():
    projection = {'results': 1, 'experiment': 1, 'status': 1}

    trials = Database().read("trials", {}, projection)
    return set(Trial.build(trials))


def build_experiment_tree(node, experiments, trials):
    return {node['name']: _build_experiment_tree(node, experiments, trials)}


def _build_experiment_tree(node, experiments, trials):
    children = {}
    trials = pop_trials(node['_id'], trials)

    for experiment in experiments:

        if (not experiment.get('refers') or
                experiment['refers']['parent_id'] != node['_id'] or
                experiment['_id'] == node['_id']):
            continue

        children[experiment['name']] = _build_experiment_tree(experiment, experiments, trials)

    return (children, trials)


def print_status_recursively(tree, depth=0, **kwargs):
    name, (subtrees, trials) = next(iter(tree.items()))
    print_status(name, trials, recursive=False, offset=depth * 2, **kwargs)

    for name, [subtree, trials] in subtrees.items():
        print_status_recursively({name: (subtree, trials)}, depth + 1)


def print_status(name, trials, recursive, offset=0, all_trials=False):
    if all_trials:
        print_all_trials(name, trials, offset=offset)
    else:
        print_summary(name, trials, offset=offset)


def print_summary(name, trials, offset=0):
    status_dict = collections.defaultdict(list)
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
        print(("\n" + (" " * offset)).join(tabulate.tabulate(lines, headers=headers).split("\n")))
    else:
        print(" " * offset, 'empty', sep="")

    print("\n")


def print_all_trials(name, trials, offset=0):
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

    print(("\n" + (" " * offset)).join(tabulate.tabulate(lines, headers=headers).split("\n")))

    print("\n")


def pop_trials(experiment_id, trials):
    experiment_trials = set()
    for trial in trials.copy():
        if trial.experiment == experiment_id:
            trials.remove(trial)
            experiment_trials.add(trial)
    return experiment_trials
