#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.client.experiment`."""
import copy
import os
import signal
import time
from contextlib import contextmanager
from multiprocessing import Process
from threading import Thread

import pytest

from orion.client.runner import LazyWorkers, Runner
from orion.core.utils.exceptions import (
    BrokenExperiment,
    CompletedExperiment,
    InvalidResult,
    ReservationRaceCondition,
    WaitingForTrials,
)
from orion.core.worker.trial import Trial
from orion.executor.base import executor_factory
from orion.storage.base import LockAcquisitionTimeout
from orion.testing import create_experiment


def new_trial(value, sleep=0.01):
    """Generate a dummy new trial"""
    return Trial(
        params=[
            dict(name="lhs", type="real", value=value),
            dict(name="sleep", type="real", value=sleep),
        ]
    )


@contextmanager
def change_signal_handler(sig, handler):
    previous = signal.signal(sig, handler)

    yield None

    signal.signal(sig, previous)


class FakeClient:
    """Orion mock client for Runner."""

    def __init__(self, n_workers):
        self.is_done = False
        self.executor = executor_factory.create("joblib", n_workers)
        self.suggest_error = WaitingForTrials
        self.trials = []
        self.status = []
        self.working_dir = ""

    def suggest(self, pool_size=None):
        """Fake suggest."""
        if self.trials:
            return self.trials.pop()

        raise self.suggest_error

    def release(self, trial, status=None):
        """Fake release."""
        self.status.append(status)

    def observe(self, trial, value):
        """Fake observe"""
        self.status.append("completed")

    def close(self):
        self._free_executor()

    def __del__(self):
        self._free_executor()

    def _free_executor(self):
        if self.executor is not None:
            self.executor.__exit__(None, None, None)
            self.executor = None
            self.executor_owner = False


class InvalidResultClient(FakeClient):
    """Fake client that raise InvalidResult on observe"""

    def __init__(self, n_workers):
        super(InvalidResultClient, self).__init__(n_workers)
        self.trials.append(new_trial(1))

    def observe(self, trial, value):
        raise InvalidResult()


def function(lhs, sleep):
    """Simple function for testing purposes."""
    time.sleep(sleep)
    return lhs + sleep


def new_runner(idle_timeout, n_workers=2, client=None):
    """Create a new runner with a mock client."""
    if client is None:
        client = FakeClient(n_workers)

    runner = Runner(
        client=client,
        fct=function,
        pool_size=10,
        idle_timeout=idle_timeout,
        max_broken=2,
        max_trials_per_worker=2,
        trial_arg=[],
        on_error=None,
    )
    runner.stat.report()
    return runner


def function_raise_on_2(lhs, sleep):
    """Simple function for testing purposes."""

    if lhs % 2 == 1:
        raise RuntimeError()

    return lhs + sleep


def test_stop_after_max_trial_reached():
    """Check that all results are registered before exception are raised"""

    count = 10
    max_trials = 1
    workers = 2

    runner = new_runner(0.1, n_workers=workers)
    runner.max_broken = 2
    runner.max_trials_per_worker = max_trials
    client = runner.client

    client.trials.extend([new_trial(i) for i in range(count)])

    runner.run()

    status = ["completed" for i in range(max_trials)]
    assert client.status == status
    runner.client.close()


def test_interrupted_scatter_gather():
    count = 2

    runner = new_runner(2, n_workers=8)
    runner.fct = function
    client = runner.client

    client.trials.extend([new_trial(i, sleep=0.75) for i in range(count, -1, -1)])

    def interrupt():
        # this should have no impact on the runner
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

    def slow_gather():
        # Sleep until some results are ready
        time.sleep(1)
        Runner.gather(runner)

    runner.gather = slow_gather

    with pytest.raises(KeyboardInterrupt):
        start = time.time()
        Thread(target=interrupt).start()

        # Gather will wait 1 sec to execute
        # so we received the sig int very early
        # but the full gather should still execute
        runner.run()

    elapsed = time.time() - start
    assert elapsed > 1, "Keyboard interrupt got delayed until gather finished"
    status = ["completed" for i in range(count)]
    assert (
        client.status == status
    ), "Trials had time to finish because of the slow gather"
    runner.client.close()


class CustomExceptionForTest(Exception):
    pass


def test_interrupted_scatter_gather_custom_signal():
    count = 2

    runner = new_runner(2, n_workers=8)
    runner.fct = function
    client = runner.client

    def custom_handler(*args):
        raise CustomExceptionForTest()

    # add a custom signal
    with change_signal_handler(signal.SIGINT, custom_handler):
        client.trials.extend([new_trial(i, sleep=0.75) for i in range(count, -1, -1)])

        def interrupt():
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGINT)

        # Our custom signal got called
        with pytest.raises(CustomExceptionForTest):
            start = time.time()
            Thread(target=interrupt).start()

            runner.run()
    runner.client.close()


def test_interrupted_scatter_gather_custom_signal_restore():
    count = 2

    runner = new_runner(2, n_workers=8)
    runner.fct = function
    client = runner.client

    def custom_handler(*args):
        raise CustomExceptionForTest()

    # add a custom signal
    with change_signal_handler(signal.SIGINT, custom_handler):
        client.trials.extend([new_trial(i, sleep=0.75) for i in range(count, -1, -1)])

        runner.run()

        # custom signal was restored
        with pytest.raises(CustomExceptionForTest):
            os.kill(os.getpid(), signal.SIGINT)
    runner.client.close()


def test_interrupted_scatter_gather_now():
    count = 2

    runner = new_runner(2, n_workers=8)
    runner.fct = function
    client = runner.client

    client.trials.extend([new_trial(i, sleep=0.75) for i in range(count, -1, -1)])

    def interrupt():
        # this will stop the runner right now
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

        # We need the sleep here or the second SIGINT is ignored
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGINT)

    def slow_gather():
        # Sleep until some results are ready
        time.sleep(1)
        Runner.gather(runner)

    runner.gather = slow_gather

    with pytest.raises(KeyboardInterrupt):
        start = time.time()
        Thread(target=interrupt).start()

        # the two interrupts forced runner to stop right now
        runner.run()

    elapsed = time.time() - start
    assert elapsed > 0.5 and elapsed < 1, "Stopped right after the 2 interrupts"
    status = ["interrupted" for i in range(count)]
    assert client.status == status, "Trials did not have time to finish"
    runner.client.close()


failures = [
    WaitingForTrials,
    ReservationRaceCondition,
    CompletedExperiment,
    LockAcquisitionTimeout,
]


@pytest.mark.parametrize("failure", failures)
def test_suggest_failures_are_handled(failure):
    runner = new_runner(0.01, n_workers=8)
    client = runner.client
    client.suggest_error = failure

    # The Suggest exception got handled
    # instead we get a LazyWorker exception
    # because not work has been queued for some time
    with pytest.raises(LazyWorkers):
        runner.run()

    runner.client.close()


def test_multi_results_with_failure():
    """Check that all results are registered before exception are raised"""

    count = 8

    runner = new_runner(0.01, n_workers=8)
    runner.max_broken = 2
    runner.max_trials_per_worker = count
    runner.fct = function_raise_on_2
    client = runner.client

    client.trials.extend([new_trial(i) for i in range(count, -1, -1)])

    new_trials = runner.sample()
    runner.scatter(new_trials)

    assert len(new_trials) == count

    # wait for multiple future to finish
    time.sleep(1)

    with pytest.raises(BrokenExperiment):
        runner.gather()

    status = ["broken" if i % 2 == 1 else "completed" for i in range(count)]
    assert client.status == status
    runner.client.close()


def test_invalid_result_worker():
    """Worker are waiting for new trials but none can be generated."""

    client = InvalidResultClient(2)
    runner = new_runner(1, client=client)

    with pytest.raises(InvalidResult):
        runner.run()

    assert client.status[0] == "broken", "Trial should be set to broken"
    runner.client.close()


def test_idle_worker():
    """Worker are waiting for new trials but none can be generated."""
    idle_timeout = 2
    runner = new_runner(idle_timeout)

    # Nothing is pending
    # Has Remaining
    # Is not done
    #
    # but no trials can be generated for our idle workers
    start = time.time()
    with pytest.raises(LazyWorkers):
        runner.run()

    elapsed = time.time() - start
    assert int(elapsed - idle_timeout) == 0, "LazyWorkers was raised after idle_timeout"
    runner.client.close()


@pytest.mark.parametrize("method", ["scatter", "gather"])
def test_idle_worker_slow(method):
    idle_timeout = 0.5
    method_sleep = 1
    count = 5
    trials = [new_trial(i, sleep=0) for i in range(count, -1, -1)]

    runner = new_runner(idle_timeout, n_workers=8)
    runner.max_trials_per_worker = len(trials)
    client = runner.client

    client.trials.extend(trials)

    def slow_method(*args, **kwargs):
        # Sleep until some results are ready
        time.sleep(method_sleep)
        getattr(Runner, method)(runner, *args, **kwargs)

    setattr(runner, method, slow_method)

    # # Should not raise LazyWorkers
    assert runner.run() == len(trials)
    runner.client.close()


def test_pending_idle_worker():
    """No new trials can be generated but we have a pending trial so LazyWorkers is not raised."""
    idle_timeout = 1
    pop_time = 1
    runner = new_runner(idle_timeout)

    # Dummy pending that will prevent runner from
    # raising LazyWorkers
    runner.pending_trials[0] = None

    def remove_pending():
        time.sleep(pop_time)
        runner.pending_trials = dict()

    start = time.time()
    thread = Thread(target=remove_pending)
    thread.start()

    with pytest.raises(LazyWorkers):
        runner.run()

    elapsed = time.time() - start

    assert (
        int(elapsed - (pop_time + idle_timeout)) == 0
    ), "LazyWorkers was raised after pending_trials got emptied"

    runner.client.close()


def test_no_remaining_worker():
    """Runner stops if we have not more trials to run"""
    idle_timeout = 2
    pop_time = 1
    runner = new_runner(idle_timeout)

    runner.pending_trials[0] = None

    def no_more_trials():
        time.sleep(pop_time)
        runner.pending_trials = dict()
        runner.trials = 2

    start = time.time()
    thread = Thread(target=no_more_trials)
    thread.start()

    # Lazy worker is not raised because we have executed
    # the max number of trials on this worker
    runner.run()

    elapsed = time.time() - start

    assert (
        int(elapsed - pop_time) == 0
    ), "Runner terminated gracefully once max trials was reached"
    runner.client.close()


def test_is_done_worker():
    """Runner stops when the experiment is_done"""
    idle_timeout = 1
    pop_time = 1
    runner = new_runner(idle_timeout)

    runner.pending_trials[0] = None

    def set_is_done():
        time.sleep(pop_time)
        runner.pending_trials = dict()
        runner.client.is_done = True

    start = time.time()
    thread = Thread(target=set_is_done)
    thread.start()

    runner.run()

    elapsed = time.time() - start

    assert (
        int(elapsed - pop_time) == 0
    ), "Runner terminated gracefully once experiment is done"
    runner.client.close()


def test_should_sample():
    """Should sample should return the number of trial we can sample"""

    def make_runner(n_workers, max_trials_per_worker, pool_size=None):
        if pool_size is None:
            pool_size = n_workers

        return Runner(
            client=FakeClient(n_workers),
            fct=function,
            pool_size=pool_size,
            idle_timeout=1,
            max_broken=2,
            max_trials_per_worker=max_trials_per_worker,
            trial_arg=[],
            on_error=None,
        )

    assert (
        make_runner(5, 2).should_sample() == 2
    ), "5 processes but only 2 trials allowed"

    assert (
        make_runner(2, 5).should_sample() == 2
    ), "2 processes and 5 max trials allowed"

    assert (
        make_runner(5, 5, 2).should_sample() == 5
    ), "5 processes and 5 max trials allowed but pool_size is 2"

    runner = make_runner(5, 10)
    runner.trials = 4
    assert runner.should_sample() == 5, "4 trials are done. 5 free processes"

    runner = make_runner(5, 10)
    runner.trials = 8
    assert runner.should_sample() == 2, "8 trials are done. 2 remains"

    runner = make_runner(5, 10)
    runner.pending_trials = [i for i in range(3)]
    runner.trials = 2
    assert runner.should_sample() == 2, "5 trials remains, but only 2 free processes"

    runner = make_runner(2, 5)
    runner.client.is_done = True
    assert runner.should_sample() == 0, "Experiment is done, no sampling"

    runner = make_runner(2, 5)
    runner.max_broken = 2
    runner.worker_broken_trials = 2
    assert runner.should_sample() == 0, "Experiment is broken, no sampling"

    runner = make_runner(2, 5)
    runner.pending_trials = [i for i in range(2)]
    assert runner.should_sample() == 0, "All processes have tasks"

    runner = make_runner(2, 5)
    runner.trials = 5
    assert runner.should_sample() == 0, "The max number of trials was reached"
    runner.client.close()
