# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker` -- Coordination of the optimization procedure
======================================================================

.. module:: worker
   :platform: Unix
   :synopsis: Executes optimization steps and runs training experiment
      with parameter values suggested.

"""
import io
import itertools
import logging
import pprint

from orion.core.worker.consumer import Consumer
from orion.core.worker.producer import Producer
from orion.storage.base import get_storage

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
            log.info("#### Experiment has reached broken trials threshold, terminating.")
            return

        if experiment.is_done:
            break

        log.debug("#### Try to reserve a new trial to evaluate.")
        trial = reserve_trial(experiment, producer)

        if trial is not None:
            log.debug("#### Successfully reserved %s to evaluate. Consuming...", trial)
            consumer.consume(trial)

    stats = experiment.stats

    if not stats:
        log.info("No trials completed.")
        return

    best = get_storage().get_trial(uid=stats['best_trials_id'])

    stats_stream = io.StringIO()
    pprint.pprint(stats, stream=stats_stream)
    stats_string = stats_stream.getvalue()

    best_stream = io.StringIO()
    pprint.pprint(best.to_dict()['params'], stream=best_stream)
    best_string = best_stream.getvalue()

    log.info("#####  Search finished successfully  #####")
    log.info("\nRESULTS\n=======\n%s\n", stats_string)
    log.info("\nBEST PARAMETERS\n===============\n%s", best_string)
