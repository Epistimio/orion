import time
from threading import Thread

import pytest

from orion.client.runner import LazyWorkers, Runner
from orion.core.utils.exceptions import WaitingForTrials
from orion.executor.base import executor_factory


class FakeClient:
    """Orion mock client for Runner."""

    def __init__(self, n_workers):
        self.is_done = False
        self.executor = executor_factory.create("joblib", n_workers)
        self.suggest_error = WaitingForTrials

    def suggest(self, pool_size=None):
        """Fake suggest."""
        raise self.suggest_error

    def release(self, *args, **kwargs):
        """Fake release."""
        pass


def function(lhs, rhs):
    """Simple function for testing purposes."""
    return lhs + rhs


def new_runner(idle_timeout, n_workers=2):
    """Create a new runner with a mock client."""
    client = FakeClient(n_workers)
    runner = Runner(
        client=client,
        fct=function,
        pool_size=2,
        idle_timeout=idle_timeout,
        max_broken=2,
        max_trials_per_worker=2,
        trial_arg=[],
        on_error=None,
    )
    return runner


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


def test_no_remaining_worker():
    """Runner stops if we have not more trials to run"""
    idle_timeout = 1
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
    assert runner.should_sample() == 5, "5 trials are done. 5 free processes"

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
