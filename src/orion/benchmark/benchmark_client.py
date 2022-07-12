#!/usr/bin/env python
"""
Benchmark client
=================
"""
import datetime
import logging

from orion.benchmark import Benchmark, Study
from orion.benchmark.assessment.base import bench_assessment_factory
from orion.benchmark.task.base import bench_task_factory
from orion.core.io.database import DuplicateKeyError
from orion.core.utils.exceptions import NoConfigurationError
from orion.storage.base import setup_storage

logger = logging.getLogger(__name__)


def get_or_create_benchmark(
    name,
    algorithms=None,
    targets=None,
    storage=None,
    storage_instance=None,
    executor=None,
    debug=False,
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
    storage_instance: BaseStorageProtocol, optional
        An instance of a storage. This argument has precedence over `storage` argument. 
    executor: `orion.executor.base.BaseExecutor`, optional
        Executor to run the benchmark experiments
    debug: bool, optional
        If using in debug mode, the storage config is overridden with legacy:EphemeralDB.
        Defaults to False.

    Returns
    -------
    An instance of `orion.benchmark.Benchmark`
    """
    storage_config = storage
    storage = storage_instance or setup_storage(storage=storage_config, debug=debug)

    # fetch benchmark from db
    db_config = _fetch_benchmark(storage, name)

    benchmark_id = None
    input_configure = None

    if db_config:
        if algorithms or targets:
            input_benchmark = Benchmark(name, algorithms, targets)
            input_configure = input_benchmark.configuration

        benchmark_id, algorithms, targets = _resolve_db_config(db_config)

    if not algorithms or not targets:
        raise NoConfigurationError(
            "Benchmark {} does not exist in DB, "
            "algorithms and targets space was not defined.".format(name)
        )

    benchmark = _create_benchmark(
        name,
        algorithms,
        targets,
        storage=storage_config,
        executor=executor,
        storage_instance=storage,
    )

    if input_configure and input_benchmark.configuration != benchmark.configuration:
        logger.warn(
            "Benchmark with same name is found but has different configuration, "
            "which will be used for this creation.\n{}".format(benchmark.configuration)
        )

    if benchmark_id is None:
        logger.debug("Benchmark not found in DB. Now attempting registration in DB.")
        try:
            _register_benchmark(storage, benchmark)
            logger.debug("Benchmark successfully registered in DB.")
        except DuplicateKeyError:
            logger.info(
                "Benchmark registration failed. This is likely due to a race condition. "
                "Now rolling back and re-attempting building it."
            )
            benchmark.close()
            benchmark = get_or_create_benchmark(
                name,
                algorithms,
                targets,
                storage_config,
                executor,
                debug,
                storage_instance=storage_instance,
            )

    return benchmark


def _get_task(name, **kwargs):
    return bench_task_factory.create(of_type=name, **kwargs)


def _get_assessment(name, **kwargs):
    return bench_assessment_factory.create(of_type=name, **kwargs)


def _resolve_db_config(db_config):

    benchmark_id = db_config["_id"]
    algorithms = db_config["algorithms"]

    obj_targets = []
    str_targets = db_config["targets"]
    for target in str_targets:
        obj_target = {}

        assessments = target["assess"]
        obj_assessments = []
        for name, parameters in assessments.items():
            obj_assessments.append(_get_assessment(name, **parameters))
        obj_target["assess"] = obj_assessments

        tasks = target["task"]
        obj_tasks = []
        for name, parameters in tasks.items():
            obj_tasks.append(_get_task(name, **parameters))
        obj_target["task"] = obj_tasks

    obj_targets.append(obj_target)

    targets = obj_targets

    return benchmark_id, algorithms, targets


def _create_benchmark(name, algorithms, targets, storage, executor, storage_instance):

    benchmark = Benchmark(
        name, algorithms, targets, storage, executor, storage_instance=storage_instance
    )
    benchmark.setup_studies()

    return benchmark


def _create_study(benchmark, algorithms, assess, task):
    study = Study(benchmark, algorithms, assess, task)
    study.setup_experiments()

    return study


def _fetch_benchmark(storage, name):

    if name:
        configs = storage.fetch_benchmark({"name": name})
    else:
        configs = storage.fetch_benchmark({})

    if not configs:
        return {}

    return configs[0]


def _register_benchmark(storage, benchmark):
    benchmark.metadata["datetime"] = datetime.datetime.utcnow()
    config = benchmark.configuration
    # This will raise DuplicateKeyError if a concurrent experiment with
    # identical (name, metadata.user) is written first in the database.
    storage.create_benchmark(config)
