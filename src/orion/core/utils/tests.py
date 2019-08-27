# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.state` -- Encapsulate Orion state
========================================================
.. module:: state
   :platform: Unix
   :synopsis: Encapsulate orion state

"""

import datetime
import os
import tempfile

import yaml

from orion.core.io.database import Database
from orion.core.io.database.ephemeraldb import EphemeralDB
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils import SingletonAlreadyInstantiatedError
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage, Storage
from orion.storage.legacy import Legacy


def _remove(file_name):
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass


def _select(lhs, rhs):
    if lhs:
        return lhs
    return rhs


def default_datetime():
    """Return default datetime"""
    return datetime.datetime(1903, 4, 25, 0, 0, 0)


class MockDatetime(datetime.datetime):
    """Fake Datetime"""

    @classmethod
    def utcnow(cls):
        """Return our random/fixed datetime"""
        return default_datetime()


def _get_default_test_database():
    """Return default configuration for the test database"""
    _, filename = tempfile.mkstemp('orion_test')

    return {
        'storage_type': 'legacy',
        'type': 'PickledDB',
        'host': filename
    }


# pylint: disable=no-self-use,protected-access
class OrionState:
    """Setup global variables and singleton for tests.

    It swaps the singleton with `None` at startup and restores them after the tests.
    It also initializes PickleDB as the storage for testing.
    We use PickledDB as our storage mock

    Parameters
    ----------
    config: YAML
        YAML config to apply for this test

    experiments: list
        List of experiments to insert into the database

    trials: list
        List of trials to insert into the database

    workers: list
        List of workers to insert into the database

    resources: list
        List of resources to insert into the database

    database: dict
        Configuration of the underlying database

    Examples
    --------
    >>> myconfig = {...}
    >>> with OrionState(myconfig):
        ...

    """

    # TODO: Fix these singletons to remove Legacy, MongoDB, PickledDB and EphemeralDB.
    SINGLETONS = (Storage, Legacy, Database, MongoDB, PickledDB, EphemeralDB)
    singletons = {}
    database = None
    experiments = []
    trials = []
    resources = []
    workers = []
    tempfile = None

    def __init__(self, experiments=None, trials=None, workers=None, lies=None, resources=None,
                 from_yaml=None, database=None):
        if from_yaml is not None:
            with open(from_yaml) as f:
                exp_config = list(yaml.safe_load_all(f))
                experiments = exp_config[0]
                trials = exp_config[1]

        self.database_config = _select(database, _get_default_test_database())
        self.experiments = _select(experiments, [])
        self.trials = _select(trials, [])
        self.workers = _select(workers, [])
        self.resources = _select(resources, [])
        self.lies = _select(lies, [])

    def init(self):
        """Initialize environment before testing"""
        self.storage()
        self.database = get_storage()._db
        self.cleanup()
        self.load_experience_configuration()
        return self

    def get_experiment(self, name, user=None, version=None):
        """Make experiment id deterministic"""
        exp = Experiment(name, user=user, version=version)
        exp._id = name
        return exp

    def get_trial(self, index):
        """Return a Trial"""
        return Trial(**self.trials[index])

    def cleanup(self):
        """Cleanup after testing"""
        self.database.remove('experiments', {})
        self.database.remove('trials', {})

    def load_experience_configuration(self):
        """Load an example database."""
        for i, t_dict in enumerate(self.trials):
            self.trials[i] = Trial(**t_dict).to_dict()

        for i, t_dict in enumerate(self.lies):
            self.lies[i] = Trial(**t_dict).to_dict()

        self.trials.sort(key=lambda obj: int(obj['_id'], 16), reverse=True)

        for i, _ in enumerate(self.experiments):
            path = os.path.join(
                os.path.dirname(__file__),
                self.experiments[i]["metadata"]["user_script"])

            self.experiments[i]["metadata"]["user_script"] = path
            self.experiments[i]['version'] = 1
            self.experiments[i]['_id'] = i

        if self.experiments:
            self.database.write('experiments', self.experiments)
        if self.trials:
            self.database.write('trials', self.trials)
        if self.workers:
            self.database.write('workers', self.workers)
        if self.resources:
            self.database.write('resources', self.resources)
        if self.lies:
            self.database.write('lying_trials', self.lies)

    def __enter__(self):
        """Load a new database state"""
        self.tempfile = self.database_config.get('host')

        for singleton in self.SINGLETONS:
            self.new_singleton(singleton, new_value=None)

        return self.init()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup database state"""
        self.cleanup()

        for obj in self.singletons:
            self.restore_singleton(obj)

    def new_singleton(self, obj, new_value=None):
        """Replace a singleton by another value"""
        self.singletons[obj] = obj.instance
        obj.instance = new_value

    def restore_singleton(self, obj):
        """Restore a singleton to its previous value"""
        obj.instance = self.singletons.get(obj)

    def storage(self):
        """Return test storage"""
        try:
            storage_type = self.database_config.pop('storage_type', 'legacy')
            config = {
                'database': self.database_config
            }
            db = Storage(of_type=storage_type, config=config)
            self.database_config['storage_type'] = storage_type

        except SingletonAlreadyInstantiatedError:
            db = get_storage()

        except KeyError:
            print(self.database_config)
            raise

        return db
