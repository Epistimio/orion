# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker` -- Coordination of the optimization procedure
======================================================================

.. module:: worker
   :platform: Unix
   :synopsis: Executes optimization steps and runs training experiment
      with parameter values suggested.

"""
import itertools
import logging

from orion.core.utils.format_terminal import format_stats
from orion.core.worker.consumer import Consumer
from orion.core.worker.producer import Producer


log = logging.getLogger(__name__)


class WaitingForTrials(Exception):
    """Raised when no trials could be reserved after multiple retries"""
    pass


def reserve_trial(experiment, producer, _depth=1):
    """Reserve a new trial, or produce and reserve a trial if none are available."""
    trial = experiment.reserve_trial()

    if trial is None and not experiment.is_done:

        if _depth > 10:
            raise WaitingForTrials('No trials are available at the moment '
                                   'wait for current trials to finish')

        log.debug("#### Failed to pull a new trial from database.")

        log.debug("#### Fetch most recent completed trials and update algorithm.")
        producer.update()

        log.debug("#### Produce new trials.")
        producer.produce()

        return reserve_trial(experiment, producer, _depth=_depth + 1)

    return trial


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


def workon(experiment, worker_trials=None):
    """Try to find solution to the search problem defined in `experiment`."""
    producer = Producer(experiment)
    consumer = Consumer(experiment)

    log.debug("#####  Init Experiment  #####")
    try:
        iterator = range(int(worker_trials))
    except (OverflowError, TypeError):
        # When worker_trials is inf
        iterator = itertools.count()

    for _ in iterator:
        log.debug("#### Poll for experiment termination.")
        if experiment.is_broken:
            print("#### Experiment has reached broken trials threshold, terminating.")
            break

        if experiment.is_done:
            print("#####  Search finished successfully  #####")
            break

        log.debug("#### Try to reserve a new trial to evaluate.")
        trial = reserve_trial(experiment, producer)

        if trial is not None:
            log.debug("#### Successfully reserved %s to evaluate. Consuming...", trial)
            consumer.consume(trial)

    print('\n' + format_stats(experiment))

    print('\n' + COMPLETION_MESSAGE.format(experiment=experiment))
    if not experiment.is_done:
        print(NONCOMPLETED_MESSAGE.format(experiment=experiment))
