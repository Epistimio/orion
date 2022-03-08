"""Used to test instantiating a runner inside a subprocess"""
from argparse import ArgumentParser

from orion.client.runner import Runner
from orion.core.utils.exceptions import WaitingForTrials
from orion.core.worker.trial import Trial
from orion.executor.base import executor_factory

idle_timeout = 20
count = 10
n_workers = 2


parser = ArgumentParser()
parser.add_argument("--backend", type=str, default="joblib")
args = parser.parse_args()


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

    def __init__(self, n_workers):
        self.is_done = False
        self.executor = executor_factory.create(args.backend, n_workers)
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


def function(lhs, sleep):
    return lhs + sleep


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

client = runner.client

client.trials.extend([new_trial(i) for i in range(count)])

runner.run()
runner.client.close()
print("done")
