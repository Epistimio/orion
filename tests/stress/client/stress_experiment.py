#!/usr/bin/env python
"""Perform a stress tests on python API."""
import os
import random
import time
import traceback
from multiprocessing import Pool

import matplotlib.pyplot as plt
from pymongo import MongoClient

from orion.client import create_experiment
from orion.core.io.database import DatabaseTimeout
from orion.core.utils.exceptions import ReservationTimeout

DB_FILE = "stress.pkl"


def f(x, worker):
    """Sleep and return objective equal to param"""
    print(f"{worker: 6d}   {x: 5f}")
    time.sleep(max(0, random.gauss(1, 0.2)))
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
    if storage == "pickleddb":
        storage_config = {"type": "pickleddb", "host": DB_FILE}
    elif storage == "mongodb":
        storage_config = {
            "type": "mongodb",
            "name": "stress",
            "host": "mongodb://user:pass@localhost",
        }

    discrete = space_type == "discrete"
    high = size  # * 2

    return create_experiment(
        "stress-test",
        space={"x": f"uniform(0, {high}, discrete={discrete})"},
        max_trials=size,
        max_idle_time=60 * 5,
        algorithm={"random": {"seed": None if space_type == "real" else 1}},
        storage={"type": "legacy", "database": storage_config},
    )


def worker(worker_id, storage, space_type, size):
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
                trial = experiment.suggest()
            except ReservationTimeout:
                trial - None

            if trial is None:
                break

            results = f(trial.params["x"], worker_id)
            num_trials += 1
            experiment.observe(trial, results=results)

        print(f"{worker_id: 6d} leaves | is done? {experiment.is_done}")
    except DatabaseTimeout as e:
        print(f"{worker_id: 6d} timeouts and leaves")
        return num_trials
    except Exception as e:
        print(f"{worker_id: 6d} crashes")
        traceback.print_exc()
        return None

    return num_trials


def stress_test(storage, space_type, workers, size):
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
    if storage == "pickleddb":
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
    elif storage == "mongodb":
        client = MongoClient(username="user", password="pass", authSource="stress")
        database = client.stress
        database.experiments.drop()
        database.lying_trials.drop()
        database.trials.drop()
        database.workers.drop()
        database.resources.drop()
        client.close()

    print("Worker  |  Point")

    with Pool(workers) as p:
        results = p.starmap(
            worker,
            zip(
                range(workers),
                [storage] * workers,
                [space_type] * workers,
                [size] * workers,
            ),
        )

    assert (
        None not in results
    ), "A worker crashed unexpectedly. See logs for the error messages."
    assert all(n > 0 for n in results), "A worker could not execute any trial."

    if space_type in ["discrete", "real-seeded"]:
        assert sum(results) == size, results
    else:
        assert sum(results) >= size, results

    experiment = get_experiment(storage, space_type, size)

    trials = experiment.fetch_trials()

    if storage == "pickleddb":
        os.remove(DB_FILE)
    elif storage == "mongodb":
        client = MongoClient(username="user", password="pass", authSource="stress")
        database = client.stress
        database.experiments.drop()
        database.lying_trials.drop()
        database.trials.drop()
        database.workers.drop()
        database.resources.drop()
        client.close()

    return trials


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

    start_time = None
    for i, trial in enumerate(trials):
        hparams.add(trial.params["x"])
        assert trial.objective.value == trial.params["x"]
        if start_time is None:
            start_time = trial.submit_time
        x.append((trial.submit_time - start_time).total_seconds())
        y.append(i)

    if space_type in ["discrete", "real-seeded"]:
        assert len(hparams) == size
    else:
        assert len(hparams) >= size

    return x[:size], y[:size]


def benchmark(workers, size):
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
    for backend in ["mongodb", "pickleddb"]:
        for space_type in ["discrete", "real", "real-seeded"]:
            trials = stress_test(backend, space_type, workers, size)
            results[(backend, space_type)] = get_timestamps(trials, size, space_type)

    return results


def main():
    """Run all stress tests and render the plot"""
    size = 500

    num_workers = [1, 4, 16, 32, 64, 128]

    fig, axis = plt.subplots(
        len(num_workers),
        1,
        figsize=(5, 1.8 * len(num_workers)),
        gridspec_kw={"hspace": 0.01, "wspace": 0},
        sharex="col",
    )

    results = {}

    for i, workers in enumerate(num_workers):

        results[workers] = benchmark(workers, size)

        for backend in ["mongodb", "pickleddb"]:
            for space_type in ["discrete", "real", "real-seeded"]:
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
