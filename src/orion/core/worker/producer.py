"""
Produce and register samples to try
===================================

Suggest new parameter sets which optimize the objective.

"""
import logging

from orion.core.io.database import DuplicateKeyError

log = logging.getLogger(__name__)


class Producer:
    """Produce suggested sets of problem's parameter space to try out.

    It uses an `Experiment`s `BaseAlgorithm` object to observe trial results
    and suggest new trials of the parameter `Space` to be evaluated. The producer
    is the bridge between the storage and the algorithm.

    """

    def __init__(self, experiment):
        """Initialize a producer.

        :param experiment: Manager of this experiment, provides convenient
           interface for interacting with the database.
        """
        log.debug("Creating Producer object.")
        self.experiment = experiment

    def observe(self, trial):
        """Observe a trial to update algorithm's status"""
        # algorithm = self.experiment.algorithms
        # if True:
        with self.experiment.acquire_algorithm_lock() as algorithm:
            algorithm.observe([trial])

    def produce(self, pool_size, timeout=60, retry_interval=1):
        """Create and register new trials."""
        log.debug("### Algorithm attempts suggesting %s new trials.", pool_size)

        n_suggested_trials = 0
        with self.experiment.acquire_algorithm_lock(
            timeout=timeout, retry_interval=retry_interval
        ) as algorithm:
            new_trials = algorithm.suggest(pool_size)

            if not new_trials and not algorithm.is_done:
                log.info(
                    "Algo does not have more trials to sample."
                    "Waiting for current trials to finish"
                )

            for new_trial in new_trials:
                try:
                    self.experiment.register_trial(new_trial)
                    n_suggested_trials += 1
                except DuplicateKeyError:
                    log.debug("Algo suggested duplicate trial %s", new_trial)

        return n_suggested_trials
