# -*- coding: utf-8 -*-
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes
"""
Runner
======

Executes the optimization process
"""
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

import orion.core
from orion.core.utils.exceptions import (
    BrokenExperiment,
    CompletedExperiment,
    InvalidResult,
    ReservationTimeout,
    WaitingForTrials,
)
from orion.core.utils.flatten import flatten, unflatten
from orion.core.worker.consumer import ExecutionError
from orion.core.worker.trial import AlreadyReleased
from orion.executor.base import AsyncException, AsyncResult


class LazyWorkers(Exception):
    """Raised when all the workers have been idle for a given amount of time"""

    pass


def _optimize(trial, fct, trial_arg, **kwargs):
    """Execute a trial on a worker"""
    kwargs.update(flatten(trial.params))

    if trial_arg:
        kwargs[trial_arg] = trial

    return fct(**unflatten(kwargs))


log = logging.getLogger(__name__)


@contextmanager
def _timer(self, name):
    start = time.time()
    yield
    total = time.time() - start

    value = getattr(self, name)
    setattr(self, name, value + total)


@dataclass
class _Stat:
    sample: int = 0
    scatter: int = 0
    gather: int = 0

    def time(self, name):
        """Measure elapsed  time of a given block"""
        return _timer(self, name)

    def report(self):
        """Show the elapsed time of different blocks"""
        lines = [
            f"Sample  {self.sample:7.4f}",
            f"Scatter {self.scatter:7.4f}",
            f"Gather  {self.gather:7.4f}",
        ]
        return "\n".join(lines)


class Runner:
    """Run the optimization process given the current executor"""

    def __init__(
        self,
        client,
        fct,
        pool_size,
        reservation_timeout,
        max_trials_per_worker,
        max_broken,
        trial_arg,
        on_error,
        interrupt_signal_code=None,
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

        self.reservation_timeout = 0.01
        self.gather_timeout = 0.01
        self.idle_timeout = reservation_timeout

        self.worker_broken_trials = 0
        self.trials = 0
        self.futures = []
        self.pending_trials = dict()
        self.free_worker = pool_size
        self.stat = _Stat()

        if interrupt_signal_code is None:
            interrupt_signal_code = orion.core.config.worker.interrupt_signal_code

        self.interrupt_signal_code = interrupt_signal_code

    @property
    def is_done(self):
        """Returns true if the experiment has finished."""
        return self.client.is_done

    @property
    def has_remaining(self):
        """Returns true if the worker can still pick up work"""
        return (
            self.max_trials_per_worker - (self.trials - self.worker_broken_trials) > 0
        )

    @property
    def is_idle(self):
        return len(self.pending_trials) <= 0

    @property
    def running(self):
        """Returns true if we are still running trials."""
        return self.pending_trials or (self.has_remaining and not self.is_done)

    def run(self):
        """Run the optimizing process until completion.

        Returns
        -------
        the total number of trials processed

        """
        idle_time = 0

        while self.running:
            idle_start = time.time()

            if not self.is_idle:
                idle_time = 0

            # Get new trials for our free workers
            with self.stat.time("sample"):
                new_trials = self.sample()

            # Scatter the new trials to our free workers
            with self.stat.time("scatter"):
                self.scatter(new_trials)

            # Gather the results of the workers that have finished
            with self.stat.time("gather"):
                self.gather()

            if self.is_idle:
                idle_time += time.time() - idle_start

            if self.is_idle and idle_time > self.idle_timeout:
                raise LazyWorkers(f"Workers have been idle for {idle_time}")

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
            new_trials = self._suggest_trials(min(self.free_worker, remains))

        log.debug(f"Sampled new {len(new_trials)} configs")
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
        log.debug("Scheduled new trials")

    def gather(self):
        """Gather the results from each worker asynchronously"""
        results = []
        try:
            results = self.client.executor.async_get(
                self.futures, timeout=self.gather_timeout
            )
        except (KeyboardInterrupt, InvalidResult):
            self.release_all()
            raise

        to_be_raised = None
        log.debug(f"Gathered new results {len(results)}")

        # register the results
        for result in results:
            self.free_worker += 1
            trial = self.pending_trials.pop(result.future)

            if isinstance(result, AsyncResult):
                # NB: observe release the trial already
                self.client.observe(trial, result.value)
                self.trials += 1

            if isinstance(result, AsyncException):
                if (
                    isinstance(result.exception, ExecutionError)
                    and result.exception.return_code == self.interrupt_signal_code
                ):
                    to_be_raised = KeyboardInterrupt()
                    self.client.release(trial, status="interrupted")
                    continue

                if isinstance(result.exception, InvalidResult):
                    # stop the optimization process if we received `InvalidResult`
                    # as all the experiments are assumed to be returning those
                    to_be_raised = result.exception

                # Regular exception, might be caused by the choosen hyperparameters
                # themselves rather than the code in particular (like Out of Memory error
                # for big batch sizes)
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

                # if we receive too many broken trials, it might indicate the user script
                # is broken, stop the experiment and let the user investigate
                if self.worker_broken_trials >= self.max_broken:
                    raise BrokenExperiment("Worker has reached broken trials threshold")

        if to_be_raised is not None:
            log.debug("Runner was interrupted")
            self.release_all()
            raise to_be_raised

    def release_all(self):
        """Release all the trials that were reserved by this runner"""
        # Sanity check
        for _, trial in self.pending_trials.items():
            try:
                self.client.release(trial)
            except AlreadyReleased:
                pass

    def _suggest_trials(self, count):
        """Suggest a bunch of trials to be dispatched to the workers"""
        trials = []
        while True:
            try:
                trial = self.client.suggest(timeout=self.reservation_timeout)
                trials.append(trial)

                if count is not None and len(trials) == count:
                    break

            # non critical errors
            except WaitingForTrials:
                break
            except ReservationTimeout:
                break
            except CompletedExperiment:
                break

        return trials
