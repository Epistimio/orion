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

from orion.core.io.database import Database
from orion.core.worker.consumer import Consumer
from orion.core.worker.producer import Producer

log = logging.getLogger(__name__)


def reserve_trial(experiment, producer):
    """Reserve a new trial, or produce and reserve a trial if none are available."""
    trial = producer.reserve_trial(score_handle=producer.algorithm.score)

    if trial is None:
        log.debug("#### Failed to pull a new trial from database.")

        log.debug("#### Fetch most recent completed trials and update algorithm.")
        producer.update()

        log.debug("#### Produce new trials.")
        producer.produce()

        return reserve_trial(experiment, producer)

    return trial


def workon(experiment, worker_trials=None):
    """Try to find solution to the search problem defined in `experiment`."""
    producer = Producer(experiment, protocol='track:file://orion_results.json')
    consumer = Consumer(experiment)

    log.debug("#####  Init Experiment  #####")
    try:
        iterator = range(int(worker_trials))
    except (OverflowError, TypeError):
        # When worker_trials is inf
        iterator = itertools.count()

    for _ in iterator:
        log.debug("#### Poll for experiment termination.")
        if experiment.is_done:
            break

        log.debug("#### Try to reserve a new trial to evaluate.")
        trial = reserve_trial(experiment, producer)

        log.debug("#### Successfully reserved %s to evaluate. Consuming...", trial)
        consumer.consume(trial)

    stats = experiment.stats

    if not stats:
        log.info("No trials completed.")
        return

    best = Database().read('trials', {'_id': stats['best_trials_id']})[0]

    stats_stream = io.StringIO()
    pprint.pprint(stats, stream=stats_stream)
    stats_string = stats_stream.getvalue()

    best_stream = io.StringIO()
    pprint.pprint(best['params'], stream=best_stream)
    best_string = best_stream.getvalue()

    log.info("#####  Search finished successfully  #####")
    log.info("\nRESULTS\n=======\n%s\n", stats_string)
    log.info("\nBEST PARAMETERS\n===============\n%s", best_string)
