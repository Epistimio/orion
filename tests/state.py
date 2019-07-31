# -*- coding: utf-8 -*-
import os

from orion.core.io.database import Database
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage, Storage
from orion.storage.legacy import Legacy

PICKLE_DB_HARDCODED = '/tmp/unittests.pkl'


def remove(db):
    """Remove a file if it exists else do nothing"""
    try:
        os.remove(db)
    except FileNotFoundError:
        pass


def select(a, b):
    if a is None:
        return b
    return a


class OrionTestState:
    """Setup global variables and singleton for tests

    it swaps the singleton with none at startup and restore them after the tests.
    It also initialize PickleDB as the storage for testing.
    We use PickledDB as our storage mock

    Parameters
    ----------
    config: YAML
        YAML config to apply for this test

    Examples
    --------

    >>> with OrionTestState(experiments=[{'name': 'whatever'}]):
        ...

    """
    SINGLETONS = (Storage, Legacy, Database, MongoDB, PickledDB)
    singletons = {}
    database = None

    def __init__(self, experiments=None, trials=None, storage=None, workers=None, resources=None, config=None):
        self.config = select(config, {
            # define what is inside the database
            'database': {
                'experiments': select(experiments, []),
                'trials': select(trials, []),
                'workers': select(workers, []),
                'resources': select(resources, [])
            },
            # define the backend for the database
            'storage': select(storage, 'legacy')
        })

    def init(self, config):
        """Initialize environment before testing"""
        # Set the singleton database
        self.storage(self.config['storage'])
        self.database = get_storage()._db

        # Insert test data inside the database
        exp_config = config.get('database')
        if exp_config is not None:
            self.load_experience_configuration()

        return self.database

    def cleanup(self):
        """Cleanup after testing"""
        remove(PICKLE_DB_HARDCODED)

    def load_experience_configuration(self):
        """Load an example database."""
        # exp_config = list(yaml.safe_load(StringIO(experience)))

        trials = []
        for i, t_dict in enumerate(self.config['database']['trials']):
            trials.append(Trial(**t_dict).to_dict())

        for i, exp_config in enumerate(self.config['database']['experiments']):
            path = os.path.join(
                os.path.dirname(__file__),
                exp_config["metadata"]["user_script"])

            exp_config["metadata"]["user_script"] = path
            exp_config['version'] = 1

        self.database.write('trials', trials)
        self.database.write('experiments', self.config['database']['experiments'])
        self.database.write('workers', self.config['database']['workers'])
        self.database.write('resources', self.config['database']['resources'])

    def __enter__(self):
        """Initialize environment, always cleanup singletons"""
        for singleton in self.SINGLETONS:
            self.new_singleton(singleton)

        return self.init(self.config)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup environment, always restore singletons"""
        self.cleanup()

        for obj in self.singletons:
            self.restore_singleton(obj)

    def new_singleton(self, obj, new_value=None):
        """Replace singleton instance by a new instance, save the old instance so we can restore it"""
        self.singletons[obj] = obj.instance
        obj.instance = new_value

    def restore_singleton(self, obj):
        """Restore previous singleton instance"""
        obj.instance = self.singletons.get(obj)

    def storage(self, storage_type):
        """Returns test storage"""
        try:
            config = {
                'database': {
                    'type': 'PickledDB',
                    'host': PICKLE_DB_HARDCODED
                }
            }
            db = Storage(storage_type, config=config)

        except ValueError:
            db = get_storage()
        return db
