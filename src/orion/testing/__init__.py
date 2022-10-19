"""
Common testing support module
=============================

Common testing support module providing defaults, functions and mocks.

"""
# pylint: disable=protected-access

import contextlib
import copy
import datetime
import os
from contextlib import contextmanager

from falcon import testing

import orion.algo.space
import orion.core.io.experiment_builder as experiment_builder
from orion.client.experiment import ExperimentClient
from orion.core.io.space_builder import SpaceBuilder
from orion.service.client.experiment import ExperimentClientREST
from orion.serving.webapi import WebApi
from orion.testing.state import OrionState

base_experiment = {
    "name": "default_name",
    "version": 0,
    "metadata": {
        "user": "default_user",
        "user_script": "abc",
        "priors": {"x": "uniform(0, 10)"},
        "datetime": "2017-11-23T02:00:00",
        "orion_version": "XYZ",
    },
    "algorithms": {"random": {"seed": 1}},
}

base_trial = {
    "experiment": "default_name",
    "status": "new",  # new, reserved, suspended, completed, broken
    "worker": None,
    "submit_time": "2017-11-23T02:00:00",
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [],
    "params": [],
}


def default_datetime():
    """Return default datetime"""
    return datetime.datetime(1903, 4, 25, 0, 0, 0)


all_status = ["completed", "broken", "reserved", "interrupted", "suspended", "new"]


def generate_trials(trial_config=None, statuses=None, exp_config=None, max_attemtps=50):
    """Generate Trials with different configurations"""
    if trial_config is None:
        trial_config = base_trial

    if statuses is None:
        statuses = all_status

    def _generate(obj, *args, value):
        if obj is None:
            return None

        obj = copy.deepcopy(obj)
        data = obj

        data[args[-1]] = value
        return obj

    new_trials = [_generate(trial_config, "status", value=s) for s in statuses]

    for i, trial in enumerate(new_trials):
        trial["submit_time"] = datetime.datetime.utcnow() + datetime.timedelta(
            seconds=i
        )
        if trial["status"] != "new":
            trial["start_time"] = datetime.datetime.utcnow() + datetime.timedelta(
                seconds=i
            )

    for i, trial in enumerate(new_trials):
        if trial["status"] == "completed":
            trial["end_time"] = datetime.datetime.utcnow() + datetime.timedelta(
                seconds=i
            )

    if exp_config:
        space = SpaceBuilder().build(exp_config["space"])
    else:
        space = SpaceBuilder().build({"x": "uniform(0, 200)"})

    # make each trial unique
    sampled = set()
    i = 0
    for trial in new_trials:
        if trial["status"] == "completed":
            trial["results"].append({"name": "loss", "type": "objective", "value": i})

        trial_stub = space.sample(seed=i)[0]
        attempts = 0
        while trial_stub.id in sampled and attempts < max_attemtps:
            trial_stub = space.sample(seed=i)[0]
            attempts += 1
            i += 1

        if attempts >= max_attemtps:
            raise RuntimeError(
                f"Cannot sample unique trials in less than {max_attemtps}"
            )

        sampled.add(trial_stub.id)
        trial["params"] = trial_stub.to_dict()["params"]

    return new_trials


def generate_benchmark_experiments_trials(
    benchmark_algorithms, experiment_config, trial_config, task_number, max_trial
):
    """Return a list of experiments and trials for a benchmark"""
    gen_exps = []
    gen_trials = []
    algo_num = len(benchmark_algorithms)
    for i in range(task_number * algo_num):
        exp = copy.deepcopy(experiment_config)
        exp["_id"] = i
        exp["name"] = f"experiment-name-{i}"
        exp["algorithms"] = benchmark_algorithms[i % algo_num]["algorithm"]
        exp["max_trials"] = max_trial
        exp["metadata"]["datetime"] = datetime.datetime.utcnow()
        gen_exps.append(exp)

        exp_trial_config = copy.deepcopy(trial_config)
        exp_trial_config["experiment"] = i
        for j, trial in enumerate(
            generate_trials(exp_trial_config, ["completed"] * max_trial)
        ):
            trial["_id"] = f"{i}_{j}"
            gen_trials.append(trial)

    return gen_exps, gen_trials


def create_study_experiments(
    state, exp_config, trial_config, algorithms, task_number, max_trial, n_workers=(1,)
):
    gen_exps, gen_trials = generate_benchmark_experiments_trials(
        algorithms, exp_config, trial_config, task_number * len(n_workers), max_trial
    )

    from orion.client.experiment import ExperimentClient
    from orion.executor.joblib_backend import Joblib

    workers = []
    for _ in range(task_number):
        for worker in n_workers:
            for _ in range(len(algorithms)):
                workers.append(worker)

    state.add_trials(*gen_trials)
    state.add_experiments(*gen_exps)

    experiments = []
    experiments_info = []
    for i in range(task_number * len(n_workers) * len(algorithms)):
        experiment = experiment_builder.build(
            f"experiment-name-{i}", storage=state.storage_config
        )

        executor = Joblib(n_workers=workers[i], backend="threading")
        client = ExperimentClient(experiment, executor=executor)
        experiments.append(client)

    for index, exp in enumerate(experiments):
        experiments_info.append((int(index / task_number), exp))

    return experiments_info


def mock_space_iterate(monkeypatch):
    """Force space to return seeds as samples instead of actually sampling

    This is useful for tests where we want to get params we can predict (0, 1, 2, ...)
    """
    sample = orion.algo.space.Space.sample

    def iterate(self, seed, *args, **kwargs):
        """Return the trials with seed value instead of sampling"""
        trials = []
        for trial in sample(self, seed=seed, *args, **kwargs):
            trials.append(
                trial.branch(params={param: seed for param in trial.params.keys()})
            )
        return trials

    monkeypatch.setattr("orion.algo.space.Space.sample", iterate)


@contextmanager
def create_experiment(
    exp_config=None, trial_config=None, statuses=None, builder=None, knowledge_base=None
):
    """Context manager for the creation of an ExperimentClient and storage init"""
    if exp_config is None:
        raise ValueError("Parameter 'exp_config' is missing")
    if trial_config is None:
        raise ValueError("Parameter 'trial_config' is missing")
    if statuses is None:
        statuses = ["new", "interrupted", "suspended", "reserved", "completed"]

    with OrionState(
        experiments=[exp_config],
        trials=generate_trials(trial_config, statuses, exp_config),
    ) as cfg:
        experiment = experiment_builder.build(
            name=exp_config["name"], storage=cfg.storage_config
        )
        if cfg.trials:
            experiment._id = cfg.trials[0]["experiment"]
        client = ExperimentClient(experiment)
        yield cfg, experiment, client

    client.close()


@contextmanager
def create_rest_experiment(exp_config, trial_config, statuses=None, builder=None):
    from orion.service.testing import get_mongo_admin, server

    if statuses is None:
        statuses = ["new", "interrupted", "suspended", "reserved", "completed"]

    with server() as (endpoint, port):
        storage = get_mongo_admin(port, owner="User1")

        with OrionState(
            experiments=[exp_config],
            trials=generate_trials(trial_config, statuses, exp_config),
            storage=storage,
        ) as cfg:

            client = ExperimentClientREST.create_experiment(
                exp_config["name"],
                storage=dict(
                    type="reststorage",
                    endpoint=endpoint,
                    token="Tok1",
                ),
            )

            # This is the REST experiment which does not have access to the storage
            # but tests are leveraging methods of the real experiment
            # experiment = client._experiment

            # We are building the experiment as if we had direct access to the mongodb
            # database, in testing we check that experiment.method == client.method
            # so we know the API match
            builder = experiment_builder.ExperimentBuilder(storage=storage)
            experiment = builder.build(exp_config["name"], version=None, branching=None)

            if cfg.trials:
                experiment._id = cfg.trials[0]["experiment"]

            yield cfg, experiment, client

    client.close()


@contextmanager
def falcon_client(exp_config=None, trial_config=None, statuses=None):
    """Context manager for the creation of an ExperimentClient and storage init"""

    with create_experiment(exp_config, trial_config, statuses) as (
        cfg,
        experiment,
        exp_client,
    ):
        falcon_client = testing.TestClient(WebApi(cfg.storage, {}))

        yield cfg, experiment, exp_client, falcon_client


class MockDatetime(datetime.datetime):
    """Fake Datetime"""

    @classmethod
    def utcnow(cls):
        """Return our random/fixed datetime"""
        return default_datetime()


@contextlib.contextmanager
def mocked_datetime(monkeypatch):
    """Make ``datetime.datetime.utcnow()`` return an arbitrary date."""
    with monkeypatch.context() as m:
        m.setattr(datetime, "datetime", MockDatetime)

        yield MockDatetime


class AssertNewFile:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            assert os.path.exists(self.filename), self.filename
            os.remove(self.filename)
