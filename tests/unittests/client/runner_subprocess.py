"""Used to test instantiating a runner inside a subprocess"""
import os
import shutil
import tempfile
from argparse import ArgumentParser

from orion.client.runner import Runner
from orion.core.utils.exceptions import WaitingForTrials
from orion.core.worker.trial import Trial
from orion.executor.base import executor_factory


def new_trial(value, sleep=0.01):
    """Generate a dummy new trial"""
    return Trial(
        params=[
            dict(name="lhs", type="real", value=value),
            dict(name="sleep", type="real", value=sleep),
        ]
    )


class FakeClient:
    """Orion mock client for Runner."""

    def __init__(self, args, n_workers):
        self.is_done = False
        self.executor = executor_factory.create(args.backend, n_workers)
        self.suggest_error = WaitingForTrials
        self.trials = []
        self.status = []
        self.working_dir = tempfile.mkdtemp(prefix="orion-test-")

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
        self._cleanup_working_dir()

    def __del__(self):
        self._free_executor()
        self._cleanup_working_dir()

    def _free_executor(self):
        if self.executor is not None:
            self.executor.__exit__(None, None, None)
            self.executor = None
            self.executor_owner = False

    def _cleanup_working_dir(self):
        if self.working_dir and os.path.isdir(self.working_dir):
            shutil.rmtree(self.working_dir, ignore_errors=True)
            self.working_dir = ""


def function(lhs, sleep):
    return lhs + sleep


def main():
    idle_timeout = 20
    count = 10
    n_workers = 2

    parser = ArgumentParser()
    parser.add_argument("--backend", type=str, default="joblib")
    args = parser.parse_args()

    client = FakeClient(args, n_workers)

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

    client = runner.client

    client.trials.extend([new_trial(i) for i in range(count)])

    runner.run()
    runner.client.close()
    print("done")


if __name__ == "__main__":
    main()
