# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.tests` -- Utils for tests
================================================
.. module:: state
   :platform: Unix
   :synopsis: Helper functions for tests

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
from orion.storage.track import Track


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
    legacy_config = {
        'database': {
            'type': 'PickledDB',
            'host': '${file}'
        }
    }

    return {
        'storage_type': 'legacy',
        'args': {
            'config': legacy_config
        }
    }


def _remove(file_name):
    if file_name is None:
        return

    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass


# pylint: disable=no-self-use,protected-access
class BaseOrionState:
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
    SINGLETONS = (Storage, Legacy, Database, MongoDB, PickledDB, EphemeralDB, Track)
    singletons = {}
    experiments = []
    trials = []
    resources = []
    workers = []

    def __init__(self, experiments=None, trials=None, workers=None, lies=None, resources=None,
                 from_yaml=None, database=None):
        if from_yaml is not None:
            with open(from_yaml) as f:
                exp_config = list(yaml.safe_load_all(f))
                experiments = exp_config[0]
                trials = exp_config[1]

        self.tempfile = None
        self.database_config = _select(database, _get_default_test_database())

        self._experiments = _select(experiments, [])
        self._trials = _select(trials, [])
        self._workers = _select(workers, [])
        self._resources = _select(resources, [])
        self._lies = _select(lies, [])

        # In case of track we also store the inserted object
        # so the user can compare in tests the different values
        self.trials = []
        self.experiments = self._experiments
        self.lies = []

    def init(self, config):
        """Initialize environment before testing"""
        self.storage(config)
        self.load_experience_configuration()
        return self

    def get_experiment(self, name, user=None, version=None):
        """Make experiment id deterministic"""
        exp = Experiment(name, user=user, version=version)
        return exp

    def get_trial(self, index):
        """Return a Trial"""
        return Trial(**self.trials[index])

    def cleanup(self):
        """Cleanup after testing"""
        _remove(self.tempfile)

    def _set_tables(self):
        self.trials = []
        self.lies = []

        for exp in self._experiments:
            get_storage().create_experiment(exp)

        for trial in self._trials:
            nt = get_storage().register_trial(Trial(**trial))
            self.trials.append(nt.to_dict())

        for lie in self._lies:
            nt = get_storage().register_lie(Trial(**lie))
            self.lies.append(nt.to_dict())

    def load_experience_configuration(self):
        """Load an example database."""
        for i, t_dict in enumerate(self._trials):
            self._trials[i] = Trial(**t_dict).to_dict()

        for i, t_dict in enumerate(self._lies):
            self._lies[i] = Trial(**t_dict).to_dict()

        self._trials.sort(key=lambda obj: int(obj['_id'], 16), reverse=True)

        for i, experiment in enumerate(self._experiments):
            if 'user_script' in experiment['metadata']:
                path = os.path.join(
                    os.path.dirname(__file__),
                    experiment["metadata"]["user_script"])
                experiment["metadata"]["user_script"] = path

            experiment['version'] = 1
            experiment['_id'] = i

        self._set_tables()

    def make_config(self):
        """Iterate over the database configuration and replace ${file}
        by the name of a temporary file
        """
        _, self.tempfile = tempfile.mkstemp('_orion_test')
        _remove(self.tempfile)

        def map_dict(fun, dictionary):
            """Return a dictionary with fun applied to each values"""
            return {k: fun(v) for k, v in dictionary.items()}

        def replace_file(v):
            """Replace `${file}` by a generated temporary file"""
            if isinstance(v, str):
                v = v.replace('${file}', self.tempfile)

            if isinstance(v, dict):
                v = map_dict(replace_file, v)

            return v

        return map_dict(replace_file, self.database_config)

    def __enter__(self):
        """Load a new database state"""
        for singleton in self.SINGLETONS:
            self.new_singleton(singleton, new_value=None)

        self.cleanup()
        return self.init(self.make_config())

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

    def storage(self, config=None):
        """Return test storage"""
        if config is None:
            return get_storage()

        try:
            storage_type = config.pop('storage_type')
            kwargs = config['args']
            db = Storage(of_type=storage_type, **kwargs)
            # storage_type = self.database_config.pop('storage_type', 'legacy')
            # db = Storage(of_type=storage_type, database=self.database_config)
            self.database_config['storage_type'] = storage_type

        except SingletonAlreadyInstantiatedError:
            db = get_storage()

        except KeyError:
            print(self.database_config)
            raise

        return db


class LegacyOrionState(BaseOrionState):
    """See :func:`~orion.utils.tests.BaseOrionState`"""

    def __init__(self, *args, **kwargs):
        super(LegacyOrionState, self).__init__(*args, **kwargs)
        self.initialized = False

    @property
    def database(self):
        """Retrieve legacy database handle"""
        return get_storage()._db

    def init(self, config):
        """Initialize environment before testing"""
        self.storage(config)
        self.initialized = True

        if hasattr(get_storage(), '_db'):
            self.database.remove('experiments', {})
            self.database.remove('trials', {})

        self.load_experience_configuration()
        return self

    def get_experiment(self, name, user=None, version=None):
        """Make experiment id deterministic"""
        exp = Experiment(name, user=user, version=version)
        exp._id = exp.name
        return exp

    def _set_tables(self):
        if self._experiments:
            self.database.write('experiments', self._experiments)
        if self._trials:
            self.database.write('trials', self._trials)
        if self._workers:
            self.database.write('workers', self._workers)
        if self._resources:
            self.database.write('resources', self._resources)
        if self._lies:
            self.database.write('lying_trials', self._lies)

        self.lies = self._lies
        self.trials = self._trials

    def cleanup(self):
        """Cleanup after testing"""
        if self.initialized:
            self.database.remove('experiments', {})
            self.database.remove('trials', {})
            _remove(self.tempfile)
        self.initialized = False


# We are faking a class constructor here
# pylint: disable=C0103
def OrionState(*args, **kwargs):
    """Build an orion state in function of the storage type"""
    database = kwargs.get('database')

    if not database or database['storage_type'] == 'legacy':
        return LegacyOrionState(*args, **kwargs)

    return BaseOrionState(*args, **kwargs)
