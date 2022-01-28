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


def test_interrupted_scatter_gather():
    count = 2

    runner = new_runner(2, n_workers=16)
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


class CustomExceptionForTest(Exception):
    pass


def test_interrupted_scatter_gather_custom_signal():
    count = 2

    runner = new_runner(2, n_workers=16)
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


def test_interrupted_scatter_gather_custom_signal_restore():
    count = 2

    runner = new_runner(2, n_workers=16)
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


def test_interrupted_scatter_gather_now():
    count = 2

    runner = new_runner(2, n_workers=16)
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


failures = [WaitingForTrials, ReservationRaceCondition, CompletedExperiment]


@pytest.mark.parametrize("failure", failures)
def test_suggest_failures_are_handled(failure):
    runner = new_runner(0.01, n_workers=16)
    client = runner.client
    client.suggest_error = failure

    # The Suggest exception got handled
    # instead we get a LazyWorker exception
    # because not work has been queued for some time
    with pytest.raises(LazyWorkers):
        runner.run()


def test_multi_results_with_failure():
    """Check that all results are registered before exception are raised"""

    count = 10

    runner = new_runner(0.01, n_workers=16)
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


def test_invalid_result_worker():
    """Worker are waiting for new trials but none can be generated."""

    client = InvalidResultClient(2)
    runner = new_runner(1, client=client)

    with pytest.raises(InvalidResult):
        runner.run()

    assert client.status[0] == "broken", "Trial should be set to broken"


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


# Those tests cover Client and Workon
#


config = dict(
    name="supernaekei",
    space={"x": "uniform(0, 200)"},
    metadata={
        "user": "tsirif",
        "orion_version": "XYZ",
        "VCS": {
            "type": "git",
            "is_dirty": False,
            "HEAD_sha": "test",
            "active_branch": None,
            "diff_sha": "diff",
        },
    },
    version=1,
    max_trials=10,
    max_broken=5,
    working_dir="",
    algorithms={"random": {"seed": 1}},
    producer={"strategy": "NoParallelStrategy"},
    refers=dict(root_id="supernaekei", parent_id=None, adapter=[]),
)


base_trial = {
    "experiment": 0,
    "status": "new",  # new, reserved, suspended, completed, broken
    "worker": None,
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [],
    "params": [],
}


def foo_1(x):
    return [dict(name="result", type="objective", value=x * 2)]


def foo_2(x, y):
    return [dict(name="result", type="objective", value=x * 2 + y)]


default_y = 2
default_z = "voila"


def foo_test_workon_hierarchical_partial_with_override(a, b):
    assert b["y"] != default_y
    assert b["z"] == default_z
    return [dict(name="result", type="objective", value=a["x"] * 2 + b["y"])]


def foo_error(x):
    raise RuntimeError()


def foo_maybe_error(x):
    foo_maybe_error.count += 1
    if foo_maybe_error.count < 5:
        raise RuntimeError()

    return [dict(name="result", type="objective", value=x * 2)]


foo_maybe_error.count = 0


def foo_trial_args(x, my_trial_arg_name):
    assert isinstance(my_trial_arg_name, Trial)
    assert my_trial_arg_name.params["x"] == x
    return [dict(name="result", type="objective", value=x * 2)]


def foo_on_error(x, q):
    if not q.empty():
        raise q.get()()

    return [dict(name="result", type="objective", value=x * 2)]


def foo_reraise(x):
    raise NotImplementedError("Do not ignore this!")


@pytest.mark.usefixtures("version_XYZ")
class TestWorkon:
    """Tests for ExperimentClient.workon"""

    def test_workon(self):
        """Verify that workon processes properly"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            client.workon(foo_1, max_trials=5)
            assert len(experiment.fetch_trials_by_status("completed")) == 5
            assert client._pacemakers == {}

    def test_workon_partial(self):
        """Verify that partial is properly passed to the function"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            client.workon(foo_2, max_trials=10, y=2)
            assert len(experiment.fetch_trials()) == 10
            assert client._pacemakers == {}

    def test_workon_partial_with_override(self):
        """Verify that partial is overriden by trial.params"""

        ext_config = copy.deepcopy(config)
        ext_config["space"]["y"] = "uniform(0, 10)"

        with create_experiment(
            exp_config=ext_config, trial_config=base_trial, statuses=[]
        ) as (cfg, experiment, client):
            default_y = 2
            assert len(experiment.fetch_trials()) == 0
            client.workon(foo_2, max_trials=1, y=default_y)
            assert len(experiment.fetch_trials_by_status("completed")) == 1
            assert experiment.fetch_trials()[0].params["y"] != 2

    def test_workon_hierarchical_partial_with_override(self):
        """Verify that hierarchical partial is overriden by trial.params"""
        default_y = 2
        default_z = "voila"

        ext_config = copy.deepcopy(config)
        ext_config["space"] = {
            "a": {"x": "uniform(0, 10, discrete=True)"},
            "b": {"y": "loguniform(1e-08, 1)"},
        }

        with create_experiment(
            exp_config=ext_config, trial_config=base_trial, statuses=[]
        ) as (cfg, experiment, client):
            assert len(experiment.fetch_trials()) == 0
            client.workon(
                foo_test_workon_hierarchical_partial_with_override,
                max_trials=5,
                b={"y": default_y, "z": default_z},
            )
            assert len(experiment.fetch_trials_by_status("completed")) == 5
            params = experiment.fetch_trials()[0].params
            assert len(params)
            assert "x" in params["a"]
            assert "y" in params["b"]

    def test_workon_max_trials(self):
        """Verify that workon stop when reaching max_trials"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            MAX_TRIALS = 5
            assert client.max_trials > MAX_TRIALS
            client.workon(foo_1, max_trials=MAX_TRIALS)
            assert len(experiment.fetch_trials_by_status("completed")) == MAX_TRIALS

    def test_workon_max_trials_resumed(self):
        """Verify that workon stop when reaching max_trials after resuming"""

        with create_experiment(
            config, base_trial, statuses=["completed", "completed"]
        ) as (
            cfg,
            experiment,
            client,
        ):
            MAX_TRIALS = 5
            assert client.max_trials > MAX_TRIALS
            assert len(experiment.fetch_trials_by_status("completed")) == 2
            client.workon(foo_1, max_trials=MAX_TRIALS)
            assert len(experiment.fetch_trials_by_status("completed")) == MAX_TRIALS

    def test_workon_max_trials_per_worker(self):
        """Verify that workon stop when reaching max_trials_per_worker"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            MAX_TRIALS = 5
            assert client.max_trials > MAX_TRIALS
            executed = client.workon(
                foo_1, max_trials=MAX_TRIALS, max_trials_per_worker=MAX_TRIALS - 1
            )
            assert executed == MAX_TRIALS - 1
            assert len(experiment.fetch_trials_by_status("completed")) == MAX_TRIALS - 1

    def test_workon_max_trials_per_worker_resumed(self):
        """Verify that workon stop when reaching max_trials_per_worker after resuming"""

        n_completed = 2
        statuses = ["completed"] * n_completed + ["new"]
        n_trials = len(statuses)

        with create_experiment(config, base_trial, statuses=statuses) as (
            cfg,
            experiment,
            client,
        ):
            MAX_TRIALS = 9
            assert client.max_trials > MAX_TRIALS
            assert len(experiment.fetch_trials_by_status("completed")) == n_completed
            executed = client.workon(
                foo_1, max_trials=MAX_TRIALS, max_trials_per_worker=2
            )
            assert executed == 2
            assert (
                len(experiment.fetch_trials_by_status("completed")) == 2 + n_completed
            )
            executed = client.workon(
                foo_1, max_trials=MAX_TRIALS, max_trials_per_worker=3
            )
            assert executed == 3
            assert (
                len(experiment.fetch_trials_by_status("completed"))
                == 3 + 2 + n_completed
            )

    def test_workon_exp_max_broken_before_worker_max_broken(self):
        """Verify that workon stop when reaching exp.max_broken"""

        MAX_TRIALS = 5
        MAX_BROKEN = 20
        test_config = copy.deepcopy(config)
        test_config["max_broken"] = MAX_BROKEN // 2

        with create_experiment(test_config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            with pytest.raises(BrokenExperiment):
                client.workon(foo_error, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN)
            n_broken_trials = len(experiment.fetch_trials_by_status("broken"))
            n_trials = len(experiment.fetch_trials())
            assert n_broken_trials == MAX_BROKEN // 2
            assert n_trials - n_broken_trials < MAX_TRIALS

    def test_workon_max_broken_all_broken(self):
        """Verify that workon stop when reaching worker's max_broken"""

        MAX_TRIALS = 5
        MAX_BROKEN = 10

        test_config = copy.deepcopy(config)
        test_config["max_broken"] = MAX_BROKEN * 2

        with create_experiment(test_config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            with pytest.raises(BrokenExperiment):
                client.workon(foo_error, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN)
            n_broken_trials = len(experiment.fetch_trials_by_status("broken"))
            n_trials = len(experiment.fetch_trials())
            assert n_broken_trials == MAX_BROKEN
            assert n_trials - n_broken_trials < MAX_TRIALS

    def test_workon_max_trials_before_max_broken(self):
        """Verify that workon stop when reaching max_trials before max_broken"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):

            MAX_TRIALS = 5
            MAX_BROKEN = 10
            assert client.max_trials > MAX_TRIALS
            client.workon(foo_maybe_error, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN)
            n_broken_trials = len(experiment.fetch_trials_by_status("broken"))
            n_trials = len(experiment.fetch_trials())
            assert n_broken_trials < MAX_BROKEN
            assert n_trials - n_broken_trials == MAX_TRIALS

    def test_workon_trial_arg(self):
        """Verify that workon pass trial when trial_arg is defined"""

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            client.workon(foo_trial_args, max_trials=5, trial_arg="my_trial_arg_name")
            assert len(experiment.fetch_trials()) == 5

    def test_workon_on_error_ignore(self):
        """Verify that workon on_error callback ignores some errors correctly"""

        def on_error(client, trial, error, worker_broken_trials):
            assert on_error.counter == worker_broken_trials
            if isinstance(error, (IndexError, IOError, AttributeError)):
                client.release(trial, "cancelled")
                return False

            on_error.counter += 1
            return True

        on_error.counter = 0

        errors = [
            IndexError,
            ValueError,
            IOError,
            NotImplementedError,
            AttributeError,
            ImportError,
        ]
        MAX_TRIALS = 5
        MAX_BROKEN = len(errors) + 1

        def make_error_queue():
            from multiprocessing import Manager

            m = Manager()
            q = m.Queue()
            for e in errors:
                q.put(e)

            return m, q

        test_config = copy.deepcopy(config)
        test_config["max_broken"] = MAX_BROKEN * 2

        manager, errors = make_error_queue()

        with manager, create_experiment(test_config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):

            client.workon(
                foo_on_error, max_trials=MAX_TRIALS, max_broken=MAX_BROKEN, q=errors
            )
            n_broken_trials = len(experiment.fetch_trials_by_status("broken"))
            n_trials = len(experiment.fetch_trials())
            assert n_broken_trials == MAX_BROKEN - 1
            assert n_trials - n_broken_trials == MAX_TRIALS

    def test_workon_on_error_raise(self):
        """Verify that workon on_error callback can raise and stop iteration"""

        def on_error(client, trial, error, worker_broken_trials):
            raise error

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            with pytest.raises(NotImplementedError) as exc:
                client.workon(
                    foo_reraise, max_trials=5, max_broken=5, on_error=on_error
                )

            assert exc.match("Do not ignore this!")

    def test_parallel_workers(self, monkeypatch):
        """Test parallel execution with joblib"""

        with create_experiment(exp_config=config, trial_config={}, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):

            with client.tmp_executor("joblib", n_workers=5, backend="threading"):
                trials = client.workon(foo_1, max_trials=5, n_workers=2)

            # Because we use 2 workers to complete 5 trials
            # at some point we are waiting for one worker to finish
            # instead of keeping that worker idle we queue another
            # so in case of failure we have a backup worker ready
            assert trials == 6

            with client.tmp_executor("joblib", n_workers=5, backend="threading"):
                trials = client.workon(foo_1, max_trials=5, n_workers=3)

            # we are already done
            assert trials == 0
