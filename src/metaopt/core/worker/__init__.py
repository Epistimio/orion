# -*- coding: utf-8 -*-
"""
:mod:`metaopt.core.worker` -- Coordination of the optimization procedure
========================================================================

.. module:: worker
   :platform: Unix
   :synopsis: Executes optimization steps and runs training experiment
      with parameter values suggested.

"""
import logging

from metaopt.core.io.database import Database
from metaopt.core.worker.consumer import Consumer
from metaopt.core.worker.producer import Producer

log = logging.getLogger(__name__)


def workon(experiment):
    """Try to find solution to the search problem defined in `experiment`."""
    producer = Producer(experiment)
    consumer = Consumer(experiment)

    while not experiment.is_done:
        trial = experiment.reserve_trial(score_handle=producer.algorithm.score)

        if trial is None:
            producer.produce()
        else:
            consumer.consume(trial)

    stats = experiment.stats
    best = Database().read('trials', {'_id': stats['best_trials_id']})[0]

    log.info("\nRESULTS\n=======\n%s\n", stats)
    log.info("\nBEST PARAMETERS\n===============\n%s", best)
