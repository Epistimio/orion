#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module running the optimization command
=======================================

Gets an experiment and iterates over it until one of the exit conditions is met

"""

import logging

import orion.core
import orion.core.io.experiment_builder as experiment_builder
from orion.client.experiment import ExperimentClient
from orion.core.cli import base as cli
from orion.core.cli import evc as evc_cli
from orion.core.utils import sigterm_as_interrupt
from orion.core.utils.exceptions import (
    BrokenExperiment,
    InexecutableUserScript,
    MissingResultFile,
)
from orion.core.utils.format_terminal import format_stats
from orion.core.worker.consumer import Consumer
from orion.core.worker.producer import Producer

log = logging.getLogger(__name__)
SHORT_DESCRIPTION = "Conducts hyperparameter optimization"
DESCRIPTION = """
This command starts hyperparameter optimization process for the user-provided model using the
configured optimization algorithm and search space.
"""


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    hunt_parser = parser.add_parser(
        "hunt", help=SHORT_DESCRIPTION, description=DESCRIPTION
    )

    orion_group = cli.get_basic_args_group(
        hunt_parser, group_name="Hunt arguments", group_help=""
    )

    orion.core.config.experiment.add_arguments(
        orion_group,
        rename=dict(max_broken="--exp-max-broken", max_trials="--exp-max-trials"),
    )

    orion_group.add_argument(
        "--max-trials",
        type=int,
        metavar="#",
        help="(DEPRECATED) This argument will be removed in v0.3. Use --exp-max-trials instead",
    )

    orion_group.add_argument(
        "--init-only",
        default=False,
        action="store_true",
        help="Only create the experiment and register in database, but do not execute any trial.",
    )

    worker_args_group = hunt_parser.add_argument_group(
        "Worker arguments (optional)",
        description="Arguments to automatically resolved branching events.",
    )

    orion.core.config.worker.add_arguments(
        worker_args_group,
        rename=dict(max_broken="--worker-max-broken", max_trials="--worker-max-trials"),
    )

    evc_cli.get_branching_args_group(hunt_parser)

    cli.get_user_args_group(hunt_parser)

    hunt_parser.set_defaults(func=main)
    hunt_parser.set_defaults(help_empty=True)  # Print help if command is empty

    return hunt_parser


COMPLETION_MESSAGE = """\
Hints
=====

Info
----

To get more information on the experiment, run the command

orion info --name {experiment.name} --version {experiment.version}

"""


NONCOMPLETED_MESSAGE = """\
Status
------

To get the status of the trials, run the command

orion status --name {experiment.name} --version {experiment.version}


For a detailed view with status of each trial listed, use the argument `--all`

orion status --name {experiment.name} --version {experiment.version} --all

"""


# pylint:disable=unused-argument
def on_error(client, trial, error, worker_broken_trials):
    """If the script is not executable, don't waste time and raise right away"""
    if isinstance(error, (InexecutableUserScript, MissingResultFile)):
        raise error

    return True


# pylint:disable=too-many-arguments
def workon(
    experiment,
    n_workers=None,
    pool_size=None,
    max_trials=None,
    max_broken=None,
    max_idle_time=None,
    reservation_timeout=None,
    heartbeat=None,
    user_script_config=None,
    interrupt_signal_code=None,
    ignore_code_changes=None,
    executor=None,
    executor_configuration=None,
    idle_timeout=None,
):
    """Try to find solution to the search problem defined in `experiment`."""

    # NOTE: Remove in v0.3.0
    if max_idle_time is not None and reservation_timeout is None:
        reservation_timeout = max_idle_time

    producer = Producer(experiment)
    consumer = Consumer(
        experiment,
        user_script_config,
        interrupt_signal_code,
        ignore_code_changes,
    )

    client = ExperimentClient(experiment, producer, heartbeat=heartbeat)

    if executor is None:
        executor = orion.core.config.worker.executor

    if executor_configuration is None:
        executor_configuration = orion.core.config.worker.executor_configuration

    log.debug("Starting workers")
    with client.tmp_executor(executor, n_workers=n_workers, **executor_configuration):
        try:
            client.workon(
                consumer,
                n_workers=n_workers,
                pool_size=pool_size,
                reservation_timeout=reservation_timeout,
                max_trials_per_worker=max_trials,
                max_broken=max_broken,
                trial_arg="trial",
                on_error=on_error,
                idle_timeout=idle_timeout,
            )
        except BrokenExperiment as e:
            print(e)

    if client.is_done:
        print("Search finished successfully")

    print("\n" + format_stats(client))

    print("\n" + COMPLETION_MESSAGE.format(experiment=client))
    if not experiment.is_done:
        print(NONCOMPLETED_MESSAGE.format(experiment=client))


def main(args):
    """Build experiment and execute hunt command"""
    args["root"] = None
    args["leafs"] = []
    # TODO: simplify when parameter parsing is refactored
    experiment = experiment_builder.build_from_args(args)

    if args["init_only"]:
        return

    config = experiment_builder.get_cmd_config(args)
    worker_config = orion.core.config.worker.to_dict()
    if config.get("worker"):
        worker_config.update(config.get("worker"))

    # If EVC is not enabled, we force Consumer to ignore code changes.
    if not config["branching"].get("enable", orion.core.config.evc.enable):
        ignore_code_changes = True
    else:
        ignore_code_changes = config["branching"].get("ignore_code_changes")

    with sigterm_as_interrupt():
        workon(experiment, ignore_code_changes=ignore_code_changes, **worker_config)
