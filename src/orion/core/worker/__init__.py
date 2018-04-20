# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker` -- Coordination of the optimization procedure
======================================================================

.. module:: worker
   :platform: Unix
   :synopsis: Executes optimization steps and runs training experiment
      with parameter values suggested.

"""
import logging

from orion.core.io.database import Database
from orion.core.worker.consumer import Consumer
from orion.core.worker.producer import Producer

log = logging.getLogger(__name__)


def workon(experiment):
    """Try to find solution to the search problem defined in `experiment`."""
    producer = Producer(experiment)
    consumer = Consumer(experiment)

    log.debug("#####  Init Experiment  #####")
    while True:
        log.debug("#### Try to reserve a new trial to evaluate.")
        trial = experiment.reserve_trial(score_handle=producer.algorithm.score)

        if trial is None:
            log.debug("#### Failed to pull a new trial from database.")

            log.debug("#### Fetch most recent completed trials and update algorithm.")
            producer.update()

            log.debug("#### Poll for experiment termination.")
            if experiment.is_done:
                break

            log.debug("#### Produce new trials.")
            producer.produce()

        else:
            log.debug("#### Successfully reserved %s to evaluate. Consuming...", trial)
            consumer.consume(trial)

    stats = experiment.stats
    best = Database().read('trials', {'_id': stats['best_trials_id']})[0]

    log.info("#####  Search finished successfully  #####")
    log.info("\nRESULTS\n=======\n%s\n", stats)
    log.info("\nBEST PARAMETERS\n===============\n%s", best)
