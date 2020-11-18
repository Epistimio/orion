import itertools
import datetime
import importlib

from orion.benchmark import Benchmark
from orion.benchmark import Study

from orion.storage.base import get_storage, setup_storage


def create_benchmark(name, algorithms=None, targets=None):
    setup_storage()
    # fetch benchmark from db
    db_config = _fetch_benchmark(name)

    benchmark_id = None
    if db_config:
        benchmark_id, algorithms, targets = _resolve_db_config(db_config[0])

    benchmark = _create_benchmark(name, algorithms, targets)

    if benchmark_id is None:
        # persist benchmark into db
        _register_benchmark(benchmark)

    return benchmark


def get_benchmarks(name=None):
    setup_storage()
    db_configs = _fetch_benchmark(name)

    for config in db_configs:
        del config['targets']

    from tabulate import tabulate
    table = tabulate(db_configs, headers='keys', tablefmt='grid', stralign='center',
                     numalign='center')
    print(table)


def _resolve_db_config(db_config):

    benchmark_id = db_config['_id']
    algorithms = db_config['algorithms']

    obj_targets = []
    str_targets = db_config['targets']
    for target in str_targets:
        obj_target = {}

        assessments = target['assess']
        obj_assessments = []
        for assessment in assessments:
            assess_cls = list(assessment.keys())[0]
            assess_cls = assess_cls.replace('-', '.')
            assess_cfg = list(assessment.values())[0]
            mod_str, _sep, class_str = assess_cls.rpartition('.')

            module = importlib.import_module(mod_str)
            assess_class = getattr(module, class_str)
            obj_assessments.append(assess_class(**assess_cfg))
        obj_target['assess'] = obj_assessments

        tasks = target['task']
        obj_tasks = []
        for task in tasks:
            task_cls = list(task.keys())[0]
            task_cls = task_cls.replace('-', '.')
            task_cfg = list(task.values())[0]
            mod_str, _sep, class_str = task_cls.rpartition('.')

            module = importlib.import_module(mod_str)
            task_class = getattr(module, class_str)
            obj_tasks.append(task_class(**task_cfg))
        obj_target['task'] = obj_tasks

    obj_targets.append(obj_target)

    targets = obj_targets

    return benchmark_id, algorithms, targets


def _create_benchmark(name, algorithms, targets):

    benchmark = Benchmark(name, algorithms, targets)
    benchmark.setup_studies()

    return benchmark


def _create_study(benchmark, algorithms, assess, task):
    study = Study(benchmark, algorithms, assess, task)
    study.setup_experiments()

    return study


def _fetch_benchmark(name):

    if name:
        configs = get_storage().fetch_benchmark({'name': name})
    else:
        configs = get_storage().fetch_benchmark({})

    if not configs:
        return {}

    return configs


def _register_benchmark(benchmark):
    benchmark.metadata['datetime'] = datetime.datetime.utcnow()
    config = benchmark.configuration
    # This will raise DuplicateKeyError if a concurrent experiment with
    # identical (name, metadata.user) is written first in the database.
    get_storage().create_benchmark(config)

