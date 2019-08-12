# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.state` -- Encapsulate Orion state
=================================================================
.. module:: state
   :platform: Unix
   :synopsis: Encapsulate orion state

"""

import os
import yaml

from orion.core.io.database import Database
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils import SingletonAlreadyInstantiatedError
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage, Storage
from orion.storage.legacy import Legacy


PICKLE_DB_HARDCODED = '/tmp/independent.pkl'


def _remove(file_name):
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass


def _select(a, b):
    if a:
        return a
    return b


class OrionState:
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
    experiments = []
    trials = []
    resources = []
    workers = []

    def __init__(self, experiments=None, trials=None, workers=None, resources=None, from_yaml=None):
        if from_yaml is not None:
            with open(from_yaml) as f:
                exp_config = list(yaml.safe_load_all(f))
                experiments = exp_config[0]
                trials = exp_config[1]

        self.experiments = _select(experiments, [])
        self.trials = _select(trials, [])
        self.workers = _select(workers, [])
        self.resources = _select(resources, [])

    def init(self):
        """Initialize environment before testing"""
        self.storage()
        self.database = get_storage()._db
        self.load_experience_configuration()

        return self

    def get_experiment(self, name, user=None, version=None):
        """Make experiment id deterministic"""
        exp = Experiment(name, user=user, version=version)
        exp._id = name
        return exp

    def cleanup(self):
        """Cleanup after testing"""
        _remove(PICKLE_DB_HARDCODED)

    def load_experience_configuration(self):
        """Load an example database."""
        for i, t_dict in enumerate(self.trials):
            self.trials[i] = Trial(**t_dict).to_dict()

        # sort by submit time like the protocol does
        self.trials.sort(key=lambda obj: obj['submit_time'], reverse=True)

        for i, _ in enumerate(self.experiments):
            path = os.path.join(
                os.path.dirname(__file__),
                self.experiments[i]["metadata"]["user_script"])

            self.experiments[i]["metadata"]["user_script"] = path
            self.experiments[i]['version'] = 1
            self.experiments[i]['_id'] = i

        self.database.write('experiments', self.experiments)
        self.database.write('trials', self.trials)
        self.database.write('workers', self.workers)
        self.database.write('resources', self.resources)

    def __enter__(self):
        """Load a new database state"""
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

    def storage(self, storage_type='legacy'):
        """Return test storage"""
        try:
            config = {
                'database': {
                    'type': 'PickledDB',
                    'host': PICKLE_DB_HARDCODED
                }
            }
            db = Storage(of_type=storage_type, config=config)

        except SingletonAlreadyInstantiatedError:
            db = get_storage()

        return db
