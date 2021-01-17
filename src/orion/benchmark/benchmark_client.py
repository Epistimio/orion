#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.benchmark.client` -- Benchmark client
================================================================

.. module:: client
   :platform: Unix
   :synopsis: Client to use Orion benchmark.

"""
import datetime
import importlib

from orion.benchmark import Benchmark, Study
from orion.core.utils.exceptions import NoConfigurationError
from orion.storage.base import get_storage, setup_storage


def get_or_create_benchmark(
    name, algorithms=None, targets=None, storage=None, debug=False
):
    """
    Create or get a benchmark object.

    Parameters
    ----------
    name: str
        Name of the benchmark
    algorithms: list, optional
        Algorithms used for benchmark, each algorithm can be a string or dict.
    targets: list, optional
        Targets for the benchmark, each target will be a dict with two keys.

        assess: list
            Assessment objects
        task: list
            Task objects
    storage: dict, optional
        Configuration of the storage backend.
    debug: bool, optional
        If using in debug mode, the storage config is overrided with legacy:EphemeralDB.
        Defaults to False.

    Returns
    -------
    An instance of `orion.benchmark.Benchmark`
    """
    setup_storage(storage=storage, debug=debug)

    # fetch benchmark from db
    db_config = _fetch_benchmark(name)

    benchmark_id = None

    if db_config:
        benchmark_id, algorithms, targets = _resolve_db_config(db_config)

    if not algorithms or not targets:
        raise NoConfigurationError(
            "Benchmark {} does not exist in DB, "
            "algorithms and targets space was not defined.".format(name)
        )

    benchmark = _create_benchmark(name, algorithms, targets)

    if benchmark_id is None:
        # persist benchmark into db
        _register_benchmark(benchmark)

    return benchmark


def _resolve_db_config(db_config):

    benchmark_id = db_config["_id"]
    algorithms = db_config["algorithms"]

    obj_targets = []
    str_targets = db_config["targets"]
    for target in str_targets:
        obj_target = {}

        assessments = target["assess"]
        obj_assessments = []
        for assessment in assessments:
            assess_cls = list(assessment.keys())[0]
            assess_cls = assess_cls.replace("-", ".")
            assess_cfg = list(assessment.values())[0]
            mod_str, _sep, class_str = assess_cls.rpartition(".")

            module = importlib.import_module(mod_str)
            assess_class = getattr(module, class_str)
            obj_assessments.append(assess_class(**assess_cfg))
        obj_target["assess"] = obj_assessments

        tasks = target["task"]
        obj_tasks = []
        for task in tasks:

            task_cls = list(task.keys())[0]
            task_cls = task_cls.replace("-", ".")
            task_cfg = list(task.values())[0]
            mod_str, _sep, class_str = task_cls.rpartition(".")

            module = importlib.import_module(mod_str)
            task_class = getattr(module, class_str)
            obj_tasks.append(task_class(**task_cfg))
        obj_target["task"] = obj_tasks

    obj_targets.append(obj_target)

    targets = obj_targets

    return benchmark_id, algorithms, targets


def _create_benchmark(name, algorithms, targets):

    # _instantiate_algo(experiment.space, kwargs.get('algorithms'))

    benchmark = Benchmark(name, algorithms, targets)
    benchmark.setup_studies()

    return benchmark


def _create_study(benchmark, algorithms, assess, task):
    study = Study(benchmark, algorithms, assess, task)
    study.setup_experiments()

    return study


def _fetch_benchmark(name):

    if name:
        configs = get_storage().fetch_benchmark({"name": name})
    else:
        configs = get_storage().fetch_benchmark({})

    if not configs:
        return {}

    return configs[0]


def _register_benchmark(benchmark):
    benchmark.metadata["datetime"] = datetime.datetime.utcnow()
    config = benchmark.configuration
    # This will raise DuplicateKeyError if a concurrent experiment with
    # identical (name, metadata.user) is written first in the database.
    get_storage().create_benchmark(config)
