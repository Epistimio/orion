#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for unittests and functional tests."""
import copy
import os

from pymongo import MongoClient
import pytest
import yaml

from orion.algo.base import (BaseAlgorithm, OptimizationAlgorithm)
import orion.core.cli
from orion.core.io.database import Database
import orion.core.io.experiment_builder as experiment_builder
import orion.core.utils.backward as backward
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage


class DumbAlgo(BaseAlgorithm):
    """Stab class for `BaseAlgorithm`."""

    def __init__(self, space, value=5,
                 scoring=0, judgement=None,
                 suspend=False, done=False, **nested_algo):
        """Configure returns, allow for variable variables."""
        self._times_called_suspend = 0
        self._times_called_is_done = 0
        self._num = None
        self._points = None
        self._results = None
        self._score_point = None
        self._judge_point = None
        self._measurements = None
        super(DumbAlgo, self).__init__(space, value=value,
                                       scoring=scoring, judgement=judgement,
                                       suspend=suspend,
                                       done=done,
                                       **nested_algo)

    def suggest(self, num=1):
        """Suggest based on `value`."""
        self._num = num
        return [self.value] * num

    def observe(self, points, results):
        """Log inputs."""
        self._points = points
        self._results = results

    def score(self, point):
        """Log and return stab."""
        self._score_point = point
        return self.scoring

    def judge(self, point, measurements):
        """Log and return stab."""
        self._judge_point = point
        self._measurements = measurements
        return self.judgement

    @property
    def should_suspend(self):
        """Cound how many times it has been called and return `suspend`."""
        self._times_called_suspend += 1
        return self.suspend

    @property
    def is_done(self):
        """Cound how many times it has been called and return `done`."""
        self._times_called_is_done += 1
        return self.done


# Hack it into being discoverable
OptimizationAlgorithm.types.append(DumbAlgo)
OptimizationAlgorithm.typenames.append(DumbAlgo.__name__.lower())


@pytest.fixture(scope='session')
def dumbalgo():
    """Return stab algorithm class."""
    return DumbAlgo


@pytest.fixture()
def exp_config():
    """Load an example database."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))

    for config in exp_config[0]:
        backward.populate_space(config)

    return exp_config


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database, db_instance):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.lying_trials.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()


@pytest.fixture()
def db_instance(null_db_instances):
    """Create and save a singleton database instance."""
    try:
        db = Database(of_type='MongoDB', name='orion_test',
                      username='user', password='pass')
    except ValueError:
        db = Database()

    return db


@pytest.fixture
def only_experiments_db(clean_db, database, exp_config):
    """Clean the database and insert only experiments."""
    database.experiments.insert_many(exp_config[0])


def ensure_deterministic_id(name, db_instance, version=1, update=None):
    """Change the id of experiment to its name."""
    experiment = db_instance.read('experiments', {'name': name, 'version': version})[0]
    db_instance.remove('experiments', {'_id': experiment['_id']})
    _id = name + "_" + str(version)
    experiment['_id'] = _id

    if experiment['refers']['parent_id'] is None:
        experiment['refers']['root_id'] = _id

    if update is not None:
        experiment.update(update)

    db_instance.write('experiments', experiment)


# Experiments combinations fixtures
@pytest.fixture
def one_experiment(monkeypatch, db_instance):
    """Create an experiment without trials."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    name = 'test_single_exp'
    orion.core.cli.main(['init_only', '-n', name,
                         './black_box.py', '--x~uniform(0,1)'])
    ensure_deterministic_id(name, db_instance)
    return get_storage().fetch_experiments({'name': name})[0]


@pytest.fixture
def one_experiment_changed_vcs(one_experiment):
    """Create an experiment without trials."""
    experiment = experiment_builder.build(name=one_experiment['name'])

    experiment.metadata['VCS'] = {
        'type': 'git', 'is_dirty': False, 'HEAD_sha': 'new', 'active_branch': 'master',
        'diff_sha': None}

    get_storage().update_experiment(experiment, metadata=experiment.metadata)


@pytest.fixture
def one_experiment_no_version(monkeypatch, one_experiment):
    """Create an experiment without trials."""
    one_experiment['name'] = one_experiment['name'] + '-no-version'
    one_experiment.pop('version')

    def fetch_without_version(query, selection=None):
        if query.get('name') == one_experiment['name'] or query == {}:
            return [copy.deepcopy(one_experiment)]

        return []

    monkeypatch.setattr(get_storage(), 'fetch_experiments', fetch_without_version)

    return one_experiment


@pytest.fixture
def with_experiment_using_python_api(monkeypatch, one_experiment):
    """Create an experiment without trials."""
    experiment = experiment_builder.build(name='from-python-api', space={'x': 'uniform(0, 10)'})

    return experiment


@pytest.fixture
def broken_refers(one_experiment, db_instance):
    """Create an experiment with broken refers."""
    ensure_deterministic_id('test_single_exp', db_instance, update=dict(refers={'oups': 'broken'}))


@pytest.fixture
def single_without_success(one_experiment):
    """Create an experiment without a succesful trial."""
    statuses = list(Trial.allowed_stati)
    statuses.remove('completed')

    exp = experiment_builder.build(name='test_single_exp')
    x = {'name': '/x', 'type': 'real'}

    x_value = 0
    for status in statuses:
        x['value'] = x_value
        trial = Trial(experiment=exp.id, params=[x], status=status)
        x_value += 1
        Database().write('trials', trial.to_dict())


@pytest.fixture
def single_with_trials(single_without_success):
    """Create an experiment with all types of trials."""
    exp = experiment_builder.build(name='test_single_exp')

    x = {'name': '/x', 'type': 'real', 'value': 100}
    results = {"name": "obj", "type": "objective", "value": 0}
    trial = Trial(experiment=exp.id, params=[x], status='completed', results=[results])
    Database().write('trials', trial.to_dict())


@pytest.fixture
def two_experiments(monkeypatch, db_instance):
    """Create an experiment and its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_double_exp',
                         './black_box.py', '--x~uniform(0,1)'])
    ensure_deterministic_id('test_double_exp', db_instance)

    orion.core.cli.main(['init_only', '-n', 'test_double_exp',
                         '--branch-to', 'test_double_exp_child', './black_box.py',
                         '--x~+uniform(0,1,default_value=0)', '--y~+uniform(0,1,default_value=0)'])
    ensure_deterministic_id('test_double_exp_child', db_instance)


@pytest.fixture
def family_with_trials(two_experiments):
    """Create two related experiments with all types of trials."""
    exp = experiment_builder.build(name='test_double_exp')
    exp2 = experiment_builder.build(name='test_double_exp_child')
    x = {'name': '/x', 'type': 'real'}
    y = {'name': '/y', 'type': 'real'}

    x_value = 0
    for status in Trial.allowed_stati:
        x['value'] = x_value
        y['value'] = x_value
        trial = Trial(experiment=exp.id, params=[x], status=status)
        x['value'] = x_value
        trial2 = Trial(experiment=exp2.id, params=[x, y], status=status)
        x_value += 1
        Database().write('trials', trial.to_dict())
        Database().write('trials', trial2.to_dict())


@pytest.fixture
def unrelated_with_trials(family_with_trials, single_with_trials):
    """Create two unrelated experiments with all types of trials."""
    exp = experiment_builder.build(name='test_double_exp_child')

    Database().remove('trials', {'experiment': exp.id})
    Database().remove('experiments', {'_id': exp.id})


@pytest.fixture
def three_experiments(two_experiments, one_experiment):
    """Create a single experiment and an experiment and its child."""
    pass


@pytest.fixture
def three_experiments_with_trials(family_with_trials, single_with_trials):
    """Create three experiments, two unrelated, with all types of trials."""
    pass


@pytest.fixture
def three_experiments_family(two_experiments, db_instance):
    """Create three experiments, one of which is the parent of the other two."""
    orion.core.cli.main(['init_only', '-n', 'test_double_exp',
                         '--branch-to', 'test_double_exp_child2', './black_box.py',
                         '--x~+uniform(0,1,default_value=0)', '--z~+uniform(0,1,default_value=0)'])
    ensure_deterministic_id('test_double_exp_child2', db_instance)


@pytest.fixture
def three_family_with_trials(three_experiments_family, family_with_trials):
    """Create three experiments, all related, two direct children, with all types of trials."""
    exp = experiment_builder.build(name='test_double_exp_child2')
    x = {'name': '/x', 'type': 'real'}
    z = {'name': '/z', 'type': 'real'}

    x_value = 0
    for status in Trial.allowed_stati:
        x['value'] = x_value
        z['value'] = x_value * 100
        trial = Trial(experiment=exp.id, params=[x, z], status=status)
        x_value += 1
        Database().write('trials', trial.to_dict())


@pytest.fixture
def three_experiments_family_branch(two_experiments, db_instance):
    """Create three experiments, each parent of the following one."""
    orion.core.cli.main(['init_only', '-n', 'test_double_exp_child',
                         '--branch-to', 'test_double_exp_grand_child', './black_box.py',
                         '--x~+uniform(0,1,default_value=0)', '--y~uniform(0,1,default_value=0)',
                         '--z~+uniform(0,1,default_value=0)'])
    ensure_deterministic_id('test_double_exp_grand_child', db_instance)


@pytest.fixture
def three_family_branch_with_trials(three_experiments_family_branch, family_with_trials):
    """Create three experiments, all related, one child and one grandchild,
    with all types of trials.

    """
    exp = experiment_builder.build(name='test_double_exp_grand_child')
    x = {'name': '/x', 'type': 'real'}
    y = {'name': '/y', 'type': 'real'}
    z = {'name': '/z', 'type': 'real'}

    x_value = 0
    for status in Trial.allowed_stati:
        x['value'] = x_value
        y['value'] = x_value * 10
        z['value'] = x_value * 100
        trial = Trial(experiment=exp.id, params=[x, y, z], status=status)
        x_value += 1
        Database().write('trials', trial.to_dict())


@pytest.fixture
def two_experiments_same_name(one_experiment, db_instance):
    """Create two experiments with the same name but different versions."""
    orion.core.cli.main(['init_only', '-n', 'test_single_exp',
                         './black_box.py', '--x~uniform(0,1)', '--y~+normal(0,1)'])
    ensure_deterministic_id('test_single_exp', db_instance, version=2)


@pytest.fixture
def three_experiments_family_same_name(two_experiments_same_name, db_instance):
    """Create three experiments, two of them with the same name but different versions and one
    with a child.
    """
    orion.core.cli.main(['init_only', '-n', 'test_single_exp', '-v', '1', '-b',
                         'test_single_exp_child', './black_box.py', '--x~uniform(0,1)',
                         '--y~+normal(0,1)'])
    ensure_deterministic_id('test_single_exp_child', db_instance)


@pytest.fixture
def three_experiments_branch_same_name(two_experiments_same_name, db_instance):
    """Create three experiments, two of them with the same name but different versions and last one
    with a child.
    """
    orion.core.cli.main(['init_only', '-n', 'test_single_exp', '-b',
                         'test_single_exp_child', './black_box.py', '--x~uniform(0,1)',
                         '--y~normal(0,1)', '--z~+normal(0,1)'])
    ensure_deterministic_id('test_single_exp_child', db_instance)


@pytest.fixture
def three_experiments_same_name(two_experiments_same_name, db_instance):
    """Create three experiments with the same name but different versions."""
    orion.core.cli.main(['init_only', '-n', 'test_single_exp',
                         './black_box.py', '--x~uniform(0,1)', '--y~normal(0,1)',
                         '--z~+normal(0,1)'])
    ensure_deterministic_id('test_single_exp', db_instance, version=3)
