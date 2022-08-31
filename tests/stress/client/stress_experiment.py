#!/usr/bin/env python
"""Perform a stress tests on python API."""
import logging
import os
import random
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager
from multiprocessing import Pool

import matplotlib.pyplot as plt
from pymongo import MongoClient

from orion.client import create_experiment
from orion.core.io.database import DatabaseTimeout
from orion.core.utils.exceptions import (
    CompletedExperiment,
    ReservationRaceCondition,
    ReservationTimeout,
    WaitingForTrials,
)

DB_FILE = "stress.pkl"
SQLITE_FILE = "db.sqlite"

ADDRESS = "192.168.0.16"

NUM_TRIALS = 1000

NUM_WORKERS = [32, 64]

LOG_LEVEL = logging.WARNING

SPACE = ["discrete", "real", "real-seeded"]
SPACE = ["real-seeded"]

# raw_worker or runner_worker
METHOD = "runner_worker"

#
#   Create the stress test user
#
# MongoDB
#
#  mongosh
#  > use admin
#  > db.createUser({
#       user: "user",
#       pwd: "pass",
#       roles: [
#           {role: 'readWrite', db: 'stress'},
#       ]
#    })
#
# PostgreSQL -- DO NOT USE THIS IN PROD - TESTING ONLY
#
#  # Switch to the user running the database
#  sudo su postgres
#
#  # open an interactive connection to the server
#  psql
#  > CREATE USER username WITH PASSWORD 'pass';
#  > CREATE ROLE orion_database_admin;
#  > CREATE ROLE orion_database_user LOGIN;
#  > GRANT orion_database_user, orion_database_user TO username;
#  >
#  > GRANT pg_write_all_data, pg_read_all_data TO username;
#  > CREATE DATABASE stress OWNER orion_database_admin;
#  \q
#
#  > \l                             # list all the database
#  > \c stress                      # Use the datatabase
#  > select * from experiments;


BACKENDS_CONFIGS = OrderedDict(
    [
        ("sqlite", {"type": "sqlalchemy", "uri": f"sqlite:///{SQLITE_FILE}"}),
        (
            "pickleddb",
            {"type": "legacy", "database": {"type": "pickleddb", "host": DB_FILE}},
        ),
        # (
        #     "postgresql",
        #     {
        #         "type": "sqlalchemy",
        #         "uri": f"postgresql://username:pass@{ADDRESS}/stress",
        #     },
        # ),
        # (
        #     "mongodb",
        #     {
        #         "type": "legacy",
        #         "database": {
        #             "type": "mongodb",
        #             "name": "stress",
        #             "host": f"mongodb://user:pass@{ADDRESS}",
        #         },
        #     },
        # ),
    ]
)


def cleanup_storage(backend):
    if backend == "pickleddb":
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)

    elif backend == "sqlite":
        if os.path.exists(SQLITE_FILE):
            os.remove(SQLITE_FILE)

    elif backend == "postgresql":
        import sqlalchemy
        from sqlalchemy.orm import Session

        from orion.storage.sql import get_tables

        engine = sqlalchemy.create_engine(
            f"postgresql://username:pass@{ADDRESS}/stress",
            echo=True,
            future=True,
        )

        # if the tables are missing just skip
        for table in get_tables():
            try:
                with Session(engine) as session:
                    session.execute(f"DROP TABLE {table.__tablename__} CASCADE;")
                    session.commit()
            except:
                traceback.print_exc()

    elif backend == "mongodb":
        client = MongoClient(
            host=ADDRESS, username="user", password="pass", authSource="stress"
        )
        database = client.stress
        database.experiments.drop()
        database.lying_trials.drop()
        database.trials.drop()
        database.workers.drop()
        database.resources.drop()
        client.close()

    else:
        raise RuntimeError("You need to cleam your backend")


def f(x, worker=-1):
    """Sleep and return objective equal to param"""
    time.sleep(max(0, random.gauss(0.1, 1)))
    return [dict(name="objective", value=x, type="objective")]


def get_experiment(storage, space_type, size):
    """Create an experiment or load from DB if already existing

    Parameters
    ----------
    storage: str
        Can be `pickleddb` or `mongodb`. A default configuration is used for each.
    space_type: str
        Can be one of
        - `discrete` Search space is discrete and limited to `max_trials`
        - `real-seeded` Search space is continuous and algos is seeded, leading to many race
          conditions while algos are sampling the same points in parallel, or
        - `real` Search space is real and algo is not seeded, leading to very few race conditions.
    size: int
        This defines `max_trials`, and the size of the search space (`uniform(0, size)`).

    """
    storage_config = BACKENDS_CONFIGS[storage]

    discrete = space_type == "discrete"
    high = size * 2

    return create_experiment(
        "stress-test",
        space={"x": f"uniform(0, {high}, discrete={discrete})"},
        max_trials=size,
        max_idle_time=60 * 5,
        algorithms={"random": {"seed": None if space_type == "real" else 1}},
        storage=storage_config,
    )


def raw_worker(worker_id, storage, space_type, size, pool_size):
    """Run trials until experiment is done

    Parameters
    ----------
    worker_id: int
        ID of the worker. This is used to distinguish logs from different workers.
    storage: str
        See `get_experiment`.
    space_type: str
        See `get_experiment`.
    size: int
        See `get_experiment`.

    """
    try:
        experiment = get_experiment(storage, space_type, size)

        assert experiment.version == 1, experiment.version

        print(f"{worker_id: 6d} enters")

        num_trials = 0
        while not experiment.is_done:
            try:
                trial = experiment.suggest(pool_size=pool_size)
            except WaitingForTrials:
                continue
            except CompletedExperiment:
                continue
            except ReservationRaceCondition:
                continue
            except ReservationTimeout:
                trial - None

            if trial is None:
                break

            x = trial.params["x"]
            results = f(x, worker_id)

            num_trials += 1
            print(f"\r    - {worker_id: 6d} {num_trials: 6d}  {x: 5.0f}", end="")
            experiment.observe(trial, results=results)

        print(f"\n{worker_id: 6d} leaves | is done? {experiment.is_done}")
    except DatabaseTimeout as e:
        print(f"\n{worker_id: 6d} timeouts and leaves")
        return num_trials
    except Exception as e:
        print(f"\n{worker_id: 6d} crashes")
        traceback.print_exc()
        return None

    return num_trials


@contextmanager
def always_clean(storage):
    cleanup_storage(storage)
    yield
    cleanup_storage(storage)


def stress_test_raw_worker(storage, space_type, workers, size, pool_size):
    """Spawn workers and run stress test with verifications

    Parameters
    ----------
    storage: str
        See `get_experiment`.
    space_type: str
        See `get_experiment`.
    workers: int
        Number of workers to run in parallel.
    size: int
        See `get_experiment`.

    Returns
    -------
    `list` of `orion.core.worker.trial.Trial`
        List of all trials at the end of the stress test

    """
    print("Worker  |  Point")

    with Pool(workers) as p:
        results = p.starmap(
            raw_worker,
            zip(
                range(workers),
                [storage] * workers,
                [space_type] * workers,
                [size] * workers,
                [pool_size] * workers,
            ),
        )

    assert (
        None not in results
    ), "A worker crashed unexpectedly. See logs for the error messages."
    assert all(n > 0 for n in results), "A worker could not execute any trial."

    assert sum(results) >= size, f"sum({results}) = {sum(results)} != {size}"

    experiment = get_experiment(storage, space_type, size)

    trials = experiment.fetch_trials_by_status("completed")

    return trials


def stress_test_runner(storage, space_type, workers, size, pool_size):
    """Spawn workers and run stress test with verifications

    Parameters
    ----------
    storage: str
        See `get_experiment`.
    space_type: str
        See `get_experiment`.
    workers: int
        Number of workers to run in parallel.
    size: int
        See `get_experiment`.

    Returns
    -------
    `list` of `orion.core.worker.trial.Trial`
        List of all trials at the end of the stress test

    """

    experiment = get_experiment(storage, space_type, size)

    experiment.workon(fct=f, n_workers=workers, pool_size=pool_size, max_trials=size)

    return experiment.fetch_trials()


def get_timestamps(trials, size, space_type):
    """Get start timestamps of the trials

    Parameters
    ----------
    trials: `list` of `orion.core.worker.trial.Trial`
        List of all trials at the end of the stress test
    space_type: str
        See `get_experiment`.
    size: int
        See `get_experiment`.

    Returns
    -------
    (`list`, `list`)
        Where rval[0] is start timestamp and rval[1] is the index of the trial.
        For instance the i-th trial timestamp is rval[0][rval[1].index(i)].

    """
    hparams = set()
    x = []
    y = []

    empty_trial = []

    start_time = None
    for i, trial in enumerate(trials):
        hparams.add(trial.params["x"])

        if trial.objective is None:
            empty_trial.append(trial)
            continue

        assert trial.objective.value == trial.params["x"]

        if start_time is None:
            start_time = trial.submit_time

        x.append((trial.submit_time - start_time).total_seconds())
        y.append(i)

    print(f"Found empty trials {empty_trial}")
    assert len(hparams) >= size, f"{len(hparams)} == {size}"
    return x[:size], y[:size]


def benchmark(workers, size, pool_size):
    """Get start timestamps of the trials

    Parameters
    ----------
    workers: int
        see: `stress_test`.
    size: int
        See `get_experiment`.

    Returns
    -------
    dict
        Dictionary containing all results of all stress tests.
        Each key is (backend, space_type). See `get_experiment` for the supported types
        of `backend`s and `space_type`s. Each values result[(backend, space_type)] is
        in the form of a (x, y) tuple, where x a the list start timestamps and y is the indexes of
        the trials. See `get_timestamps` for more details.

    """
    results = {}

    stres_test_method = None
    if METHOD == "raw_worker":
        stres_test_method = stress_test_raw_worker
    else:
        stres_test_method = stress_test_runner

    for backend in BACKENDS_CONFIGS.keys():
        for space_type in SPACE:
            print(backend, space_type)

            # Initialize the storage once before parallel work
            get_experiment(backend, space_type, size)

            with always_clean(backend):
                trials = stres_test_method(
                    backend, space_type, workers, size, pool_size
                )

                results[(backend, space_type)] = get_timestamps(
                    trials, size, space_type
                )

    return results


def main():
    """Run all stress tests and render the plot"""
    size = NUM_TRIALS

    logging.basicConfig(level=LOG_LEVEL)

    num_workers = NUM_WORKERS

    fig, axis = plt.subplots(
        len(num_workers),
        1,
        figsize=(5, 1.8 * len(num_workers)),
        gridspec_kw={"hspace": 0.01, "wspace": 0},
        sharex="col",
    )

    results = {}

    for i, workers in enumerate(num_workers):

        results[workers] = benchmark(workers, size, pool_size=workers)

        for backend in BACKENDS_CONFIGS.keys():
            for space_type in SPACE:
                x, y = results[workers][(backend, space_type)]
                axis[i].plot(x, y, label=f"{backend}-{space_type}")

    for i, workers in enumerate(num_workers):
        # We pick 'pickleddb' and discrete=True as the reference for the slowest ones
        x, y = results[min(num_workers)][("pickleddb", "discrete")]
        d_x = max(x) - min(x)
        d_y = max(y) - min(y)
        if i < len(num_workers) - 1:
            axis[i].text(
                min(x) + d_x * 0.6, min(y) + d_y * 0.1, f"{workers: 3d} workers"
            )
        else:
            axis[i].text(
                min(x) + d_x * 0.6, min(y) + d_y * 0.7, f"{workers: 3d} workers"
            )

    for i in range(len(num_workers) - 1):
        axis[i].spines["top"].set_visible(False)
        axis[i].spines["right"].set_visible(False)

    axis[-1].spines["right"].set_visible(False)
    axis[-1].spines["top"].set_visible(False)

    axis[-1].set_xlabel("Time (s)")
    axis[-1].set_ylabel("Number of trials")
    axis[-1].legend()

    plt.subplots_adjust(left=0.15, bottom=0.05, top=1, right=1)

    plt.savefig("test.png")


if __name__ == "__main__":
    main()
