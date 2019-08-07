from io import StringIO
import os
import yaml

from orion.core.io.database import Database
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage, Storage
from orion.storage.legacy import Legacy


def remove(file_name):
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass


PICKLE_DB_HARDCODED = '/tmp/database.pkl'


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

    >>> myconfig = {...}
    >>> with OrionTestState(myconfig):
        ...

    """
    SINGLETONS = (Storage, Legacy, Database, MongoDB, PickledDB)
    singletons = {}
    database = None

    # def __init__(self, experiments, trials, ...):
    #     self.config = {
    #         'database': {
    #             'experiments': experiments,
    #             'trials': trials,
    #             ...
    #         }
    #     }

    def init(self, config, storage):
        """Initialize environment before testing"""
        self.storage(storage_type=storage)
        self.database = get_storage()._db

        exp_config = config.get('database')
        if exp_config is not None:
            self.load_experience_configuration(exp_config)

        return self.database

    def cleanup(self):
        """Cleanup after testing"""
        remove(PICKLE_DB_HARDCODED)

    def load_experience_configuration(self, experience):
        """Load an example database."""
        exp_config = list(yaml.safe_load(StringIO(experience)))

        for i, t_dict in enumerate(exp_config[1]):
            exp_config[1][i] = Trial(**t_dict).to_dict()

        for i, _ in enumerate(exp_config[0]):
            path = os.path.join(
                os.path.dirname(__file__),
                exp_config[0][i]["metadata"]["user_script"])

            exp_config[0][i]["metadata"]["user_script"] = path
            exp_config[0][i]['version'] = 1

        self.database.write('experiments', exp_config[0])
        self.database.write('trials', exp_config[1])
        self.database.write('workers', exp_config[2])
        self.database.write('resources', exp_config[3])

    def __init__(self, config=None):
        self.config = config

    def __enter__(self):
        for singleton in self.SINGLETONS:
            self.new_singleton(singleton)

        return self.init(self.config)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

        for obj in self.singletons:
            self.restore_singleton(obj)

    def new_singleton(self, obj, new_value=None):
        self.singletons[obj] = obj.instance
        obj.instance = new_value

    def restore_singleton(self, obj):
        obj.instance = self.singletons.get(obj)

    def storage(self, storage_type='legacy'):
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


class TestStorage:
    def __init__(self, backend):
        self.backend = backend

    def test_create_experiment(self):
        with OrionTestState(storage=self.backend):
            get_storage().create_experiment({
                'name': 'test'
            })

    def test_update_experiment(self):
        with OrionTestState(storage=self.backend):
            get_storage().create_experiment({
                'name': 'test'
            })
