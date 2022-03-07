# -*- coding: utf-8 -*-
# pylint:disable=too-many-arguments
# pylint:disable=too-many-instance-attributes
"""
Runner
======

Executes the optimization process
"""
import logging
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass

import orion.core
from orion.core.utils import backward
from orion.core.utils.exceptions import (
    BrokenExperiment,
    CompletedExperiment,
    InvalidResult,
    LazyWorkers,
    ReservationRaceCondition,
    WaitingForTrials,
)
from orion.core.utils.flatten import flatten, unflatten
from orion.core.worker.consumer import ExecutionError
from orion.core.worker.trial import AlreadyReleased
from orion.executor.base import AsyncException, AsyncResult
from orion.storage.base import LockAcquisitionTimeout

log = logging.getLogger(__name__)


class Protected(object):
    """Prevent a signal to be raised during the execution of some code"""

    def __init__(self):
        self.signal_received = None
        self.handlers = dict()
        self.start = 0
        self.delayed = 0
        self.signal_installed = False

    def __enter__(self):
        """Override the signal handlers with our delayed handler"""
        self.signal_received = False

        try:
            self.handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self.handler)
            self.handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self.handler)
            self.signal_installed = True

        except ValueError:  # ValueError: signal only works in main thread
            log.warning(
                "SIGINT/SIGTERM protection hooks could not be installed because "
                "Runner is executing inside a thread/subprocess, results could get lost "
                "on interruptions"
            )

        return self

    def handler(self, sig, frame):
        """Register the received signal for later"""
        log.warning("Delaying signal %d to finish operations", sig)
        log.warning(
            "Press CTRL-C again to terminate the program now  (You may lose results)"
        )

        self.start = time.time()

        self.signal_received = (sig, frame)

        # if CTRL-C is pressed again the original handlers will handle it
        # and make the program stop
        self.restore_handlers()

    def restore_handlers(self):
        """Restore old signal handlers"""
        if not self.signal_installed:
            return

        signal.signal(signal.SIGINT, self.handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, self.handlers[signal.SIGTERM])

    def stop_now(self):
        """Raise the delayed signal if any or restore the old signal handlers"""

        if not self.signal_received:
            self.restore_handlers()

        else:
            self.delayed = time.time() - self.start

            log.warning("Termination was delayed by %.4f s", self.delayed)
            handler = self.handlers[self.signal_received[0]]

            if callable(handler):
                handler(*self.signal_received)

    def __exit__(self, *args):
        self.stop_now()


def _optimize(trial, fct, trial_arg, **kwargs):
    """Execute a trial on a worker"""

    kwargs.update(flatten(trial.params))

    if trial_arg:
        kwargs[trial_arg] = trial

    return fct(**unflatten(kwargs))


@dataclass
class _Stat:
    sample: int = 0
    scatter: int = 0
    gather: int = 0

    @contextmanager
    def time(self, name):
        """Measure elapsed  time of a given block"""
        start = time.time()
        yield
        total = time.time() - start

        value = getattr(self, name)
        setattr(self, name, value + total)

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
        idle_timeout,
        max_trials_per_worker,
        max_broken,
        trial_arg,
        on_error,
        interrupt_signal_code=None,
        gather_timeout=0.01,
        n_workers=None,
        **kwargs,
    ):
        self.client = client
        self.fct = fct
        self.batch_size = pool_size
        self.max_trials_per_worker = max_trials_per_worker
        self.max_broken = max_broken
        self.trial_arg = trial_arg
        self.on_error = on_error
        self.kwargs = kwargs

        self.gather_timeout = gather_timeout
        self.idle_timeout = idle_timeout

        self.worker_broken_trials = 0
        self.trials = 0
        self.futures = []
        self.pending_trials = dict()
        self.stat = _Stat()
        self.n_worker_override = n_workers

        if interrupt_signal_code is None:
            interrupt_signal_code = orion.core.config.worker.interrupt_signal_code

        self.interrupt_signal_code = interrupt_signal_code

    @property
    def free_worker(self):
        """Returns the number of free worker"""
        n_workers = self.client.executor.n_workers

        if self.n_worker_override is not None:
            n_workers = self.n_worker_override

        return max(n_workers - len(self.pending_trials), 0)

    @property
    def is_done(self):
        """Returns true if the experiment has finished."""
        return self.client.is_done

    @property
    def is_broken(self):
        """Returns true if the experiment is broken"""
        return self.worker_broken_trials >= self.max_broken

    @property
    def has_remaining(self):
        """Returns true if the worker can still pick up work"""
        return (
            self.max_trials_per_worker - (self.trials - self.worker_broken_trials) > 0
        )

    @property
    def is_idle(self):
        """Returns true if none of the workers are running a trial"""
        return len(self.pending_trials) <= 0

    @property
    def is_running(self):
        """Returns true if we are still running trials."""
        return len(self.pending_trials) > 0 or (self.has_remaining and not self.is_done)

    def run(self):
        """Run the optimizing process until completion.

        Returns
        -------
        the total number of trials processed

        """
        idle_start = time.time()
        idle_end = 0
        idle_time = 0

        while self.is_running:
            try:

                # Protected will prevent Keyboard interrupts from
                # happening in the middle of the scatter-gather process
                # that we can be sure that completed trials are observed
                with Protected():

                    # Get new trials for our free workers
                    with self.stat.time("sample"):
                        new_trials = self.sample()

                    # Scatter the new trials to our free workers
                    with self.stat.time("scatter"):
                        scattered = self.scatter(new_trials)

                    # Gather the results of the workers that have finished
                    with self.stat.time("gather"):
                        gathered = self.gather()

                    if scattered == 0 and gathered == 0 and self.is_idle:
                        idle_end = time.time()
                        idle_time += idle_end - idle_start
                        idle_start = idle_end

                        log.debug(f"Workers have been idle for {idle_time:.2f} s")
                    else:
                        idle_start = time.time()
                        idle_time = 0

                    if self.is_idle and idle_time > self.idle_timeout:
                        msg = f"Workers have been idle for {idle_time:.2f} s"

                        if self.has_remaining and not self.is_done:
                            msg = (
                                f"{msg}; worker has leg room (has_remaining: {self.has_remaining})"
                                f" and optimization is not done (is_done: {self.is_done})"
                            )

                        raise LazyWorkers(msg)

            except KeyboardInterrupt:
                self._release_all()
                raise
            except:
                self._release_all()
                raise

        return self.trials

    def should_sample(self):
        """Check if more trials could be generated"""

        if self.free_worker <= 0 or (self.is_broken or self.is_done):
            return 0

        pending = len(self.pending_trials) + self.trials
        remains = self.max_trials_per_worker - pending

        n_trial = min(self.free_worker, remains)
        should_sample_more = self.free_worker > 0 and remains > 0

        return int(should_sample_more) * n_trial

    def sample(self):
        """Sample new trials for all free workers"""
        n_trial = self.should_sample()

        if n_trial > 0:
            # the producer does the job of limiting the number of new trials
            # already no need to worry about it
            # NB: suggest reserve the trial already
            new_trials = self._suggest_trials(n_trial)
            log.debug(f"Sampled {len(new_trials)} new configs")
            return new_trials

        return []

    def scatter(self, new_trials):
        """Schedule new trials to be computed"""
        new_futures = []
        for trial in new_trials:
            backward.ensure_trial_working_dir(self.client, trial)

            future = self.client.executor.submit(
                _optimize, trial, self.fct, self.trial_arg, **self.kwargs
            )
            self.pending_trials[future] = trial
            new_futures.append(future)

        self.futures.extend(new_futures)
        log.debug("Scheduled new trials")
        return len(new_futures)

    def gather(self):
        """Gather the results from each worker asynchronously"""
        results = self.client.executor.async_get(
            self.futures, timeout=self.gather_timeout
        )

        to_be_raised = None
        log.debug(f"Gathered new results {len(results)}")
        # register the results
        # NOTE: For Ptera instrumentation
        trials = 0  # pylint:disable=unused-variable
        for result in results:
            trial = self.pending_trials.pop(result.future)

            if isinstance(result, AsyncResult):
                try:
                    # NB: observe release the trial already
                    self.client.observe(trial, result.value)
                    self.trials += 1
                    # NOTE: For Ptera instrumentation
                    trials = self.trials  # pylint:disable=unused-variable
                except InvalidResult as exception:
                    # stop the optimization process if we received `InvalidResult`
                    # as all the trials are assumed to be returning those
                    to_be_raised = exception
                    self.client.release(trial, status="broken")

            if isinstance(result, AsyncException):
                if (
                    isinstance(result.exception, ExecutionError)
                    and result.exception.return_code == self.interrupt_signal_code
                ):
                    to_be_raised = KeyboardInterrupt()
                    self.client.release(trial, status="interrupted")
                    continue

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
        if self.is_broken:
            to_be_raised = BrokenExperiment(
                "Worker has reached broken trials threshold"
            )

        if to_be_raised is not None:
            log.debug("Runner was interrupted")
            self._release_all()
            raise to_be_raised

        return len(results)

    def _release_all(self):
        """Release all the trials that were reserved by this runner.
        This is only called during exception handling to avoid retaining trials
        that cannot be retrieved anymore

        """
        # Sanity check
        for _, trial in self.pending_trials.items():
            try:
                self.client.release(trial, status="interrupted")
            except AlreadyReleased:
                pass

        self.pending_trials = dict()

    def _suggest_trials(self, count):
        """Suggest a bunch of trials to be dispatched to the workers"""
        trials = []
        for _ in range(count):
            try:
                batch_size = count if self.batch_size == 0 else self.batch_size
                trial = self.client.suggest(pool_size=batch_size)
                trials.append(trial)

            # non critical errors
            except WaitingForTrials:
                log.debug("Runner cannot sample because WaitingForTrials")
                break

            except ReservationRaceCondition:
                log.debug("Runner cannot sample because ReservationRaceCondition")
                break

            except LockAcquisitionTimeout:
                log.debug("Runner cannot sample because LockAcquisitionTimeout")
                break

            except CompletedExperiment:
                log.debug("Runner cannot sample because CompletedExperiment")
                break

        return trials
