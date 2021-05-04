# -*- coding: utf-8 -*-
"""
Coordination of the optimization procedure
==========================================

Executes optimization steps and runs training experiment with parameter values suggested.

"""
import itertools
import logging

from orion.core.utils.exceptions import (
    BrokenExperiment,
    InexecutableUserScript,
    MissingResultFile,
)
from orion.core.utils.format_terminal import format_stats
from orion.core.worker.consumer import Consumer
from orion.core.worker.producer import Producer

log = logging.getLogger(__name__)


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


def on_error(experiment_client, trial, error, worker_broken_trials):
    """If the script is not executable, don't waste time and raise right away"""
    if isinstance(error, (InexecutableUserScript, MissingResultFile)):
        raise error

    return True


# TODO: Move this to hunt cli. Move `reserve_trial` to experiment client.
def workon(
    experiment,
    max_trials=None,
    max_broken=None,
    max_idle_time=None,
    heartbeat=None,
    user_script_config=None,
    interrupt_signal_code=None,
    ignore_code_changes=None,
):
    """Try to find solution to the search problem defined in `experiment`."""
    producer = Producer(experiment, max_idle_time)
    consumer = Consumer(
        experiment,
        user_script_config,
        interrupt_signal_code,
        ignore_code_changes,
    )

    # NOTE: Temporary fix before we move workon to orion.core.cli.hunt
    from orion.client.experiment import ExperimentClient

    experiment_client = ExperimentClient(experiment, producer, heartbeat=heartbeat)

    log.debug("Starting workers")
    try:
        experiment_client.workon(
            consumer,
            max_trials=max_trials,
            max_broken=max_broken,
            trial_arg="trial",
            on_error=on_error,
        )
    except BrokenExperiment as e:
        print(e)

    if experiment.is_done:
        print("Search finished successfully")

    print("\n" + format_stats(experiment))

    print("\n" + COMPLETION_MESSAGE.format(experiment=experiment))
    if not experiment.is_done:
        print(NONCOMPLETED_MESSAGE.format(experiment=experiment))
