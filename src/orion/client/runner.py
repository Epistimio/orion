import logging

from orion.core.utils.flatten import flatten, unflatten
from orion.core.worker.trial import AlreadyReleased
from orion.executor.base import AsyncException, AsyncResult, executor_factory
from orion.core.utils.exceptions import BrokenExperiment, InvalidResult


def _optimize(trial, fct, trial_arg, **kwargs):
    """Execute a trial on a worker"""
    kwargs.update(flatten(trial.params))

    if trial_arg:
        kwargs[trial_arg] = trial

    return fct(**unflatten(kwargs))


log = logging.getLogger(__name__)


class Runner:
    """Run the optimization process given the current executor"""

    def __init__(
        self,
        client,
        fct,
        pool_size,
        max_trials_per_worker,
        max_broken,
        trial_arg,
        on_error,
        **kwargs,
    ):
        self.client = client
        self.fct = fct
        self.pool_size = pool_size
        self.max_trials_per_worker = max_trials_per_worker
        self.max_broken = max_broken
        self.trial_arg = trial_arg
        self.on_error = on_error
        self.kwargs = kwargs

        self.worker_broken_trials = 0
        self.trials = 0
        self.futures = []
        self.pending_trials = dict()
        self.free_worker = pool_size

    @property
    def is_done(self):
        """Returns true if the experiment has finished."""
        return self.client.is_done

    @property
    def has_reached_max(self):
        """Returns true if the worker has reached his max number of trials."""
        return self.trials - self.worker_broken_trials < self.max_trials_per_worker

    @property
    def running(self):
        """Returns true if we are still running trials."""
        return self.pending_trials or (not self.is_done and self.has_reached_max)

    def run(self):
        """Run the optimizing process until completion.

        Returns
        -------
        the total number of trials processed

        """
        while self.running:
            # Get new trials for our free workers
            new_trials = self.sample()

            # Scatter the new trials to our free workers
            self.scatter(new_trials)

            # Gather the results of the workers that has finished
            self.gather()

        return self.trials

    def sample(self):
        """Sample new trials for all free workers"""
        ntrials = len(self.pending_trials) + self.trials
        remains = self.max_trials_per_worker - ntrials

        # try to get more work
        new_trials = []
        if (not self.is_done) and self.free_worker > 0 and remains > 0:
            # the producer does the job of limiting the number of new trials
            # already no need to worry about it
            # NB: suggest reserve the trial already
            new_trials = self.client._suggest_trials(min(self.free_worker, remains))

        return new_trials

    def scatter(self, new_trials):
        """Schedule new trials to be computed"""
        new_futures = []
        for trial in new_trials:
            future = self.client.executor.submit(
                _optimize, trial, self.fct, self.trial_arg, **self.kwargs
            )
            self.pending_trials[future] = trial
            new_futures.append(future)

        self.free_worker -= len(new_futures)
        self.futures.extend(new_futures)

    def gather(self):
        """Gather the results from each worker asynchronously"""
        results = []
        try:
            results = self.client.executor.async_get(self.futures, timeout=0.01)
        except (KeyboardInterrupt, InvalidResult):
            self.release_all()
            raise

        # register the results
        for result in results:
            self.free_worker += 1
            trial = self.pending_trials.pop(result.future)

            if isinstance(result, AsyncResult):
                # NB: observe release the trial already
                self.client.observe(trial, result.value)
                self.trials += 1

            if isinstance(result, AsyncException):
                exception = result.exception
                self.worker_broken_trials += 1
                self.client.release(trial, status="broken")

                if self.on_error is None or self.on_error(
                    self, trial, exception, self.worker_broken_trials
                ):
                    log.error(result.traceback)

                else:
                    log.error(str(exception))
                    log.debug(result.traceback)

                if self.worker_broken_trials >= self.max_broken:
                    raise BrokenExperiment("Worker has reached broken trials threshold")

    def release_all(self):
        """Release all the trials that were reserved by this runner"""
        # Sanity check
        for _, trial in self.pending_trials.items():
            try:
                self.client.release(trial)
            except AlreadyReleased:
                pass
