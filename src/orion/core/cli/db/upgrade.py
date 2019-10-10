#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.db.upgrade` -- Module to upgrade DB schemes
================================================================

.. module:: test_db
   :platform: Unix
   :synopsis: Upgrade the scheme of the databases

"""
import argparse
import logging
import sys

from orion.core.io.database.ephemeraldb import EphemeralCollection
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.io.experiment_builder import ExperimentBuilder
import orion.core.utils.backward as backward
from orion.storage.base import get_storage
from orion.storage.legacy import Legacy


log = logging.getLogger(__name__)


# TODO: Move somewhere else to share with `db setup`.
def ask_question(question, default=None):
    """Ask a question to the user and receive an answer.

    Parameters
    ----------
    question: str
        The question to be asked.
    default: str
        The default value to use if the user enters nothing.

    Returns
    -------
    str
        The answer provided by the user.

    """
    if default is not None:
        question = question + " (default: {}) ".format(default)

    answer = input(question)

    if answer.strip() == "":
        return default

    return answer


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    upgrade_db_parser = parser.add_parser('upgrade', help='Upgrade the database scheme')

    upgrade_db_parser.add_argument('-c', '--config', type=argparse.FileType('r'),
                                   metavar='path-to-config', help="user provided "
                                   "orion configuration file")

    upgrade_db_parser.add_argument('-f', '--force', action='store_true',
                                   help="Don't prompt user")

    upgrade_db_parser.set_defaults(func=main)

    return upgrade_db_parser


def main(args):
    """Upgrade the databases for current version"""
    print("Upgrading your database may damage your data. Make sure to make a backup before the "
          "upgrade and stop any other process that may read/write the database during the upgrade.")

    if not args.get('force'):
        action = ''
        while action not in ['y', 'yes', 'no', 'n']:
            action = ask_question("Do you wish to proceed? (y/N)", "N").lower()

        if action in ['no', 'n']:
            sys.exit(0)

    experiment_builder = ExperimentBuilder()
    local_config = experiment_builder.fetch_full_config(args, use_db=False)
    local_config['protocol'] = {'type': 'legacy', 'setup': False}

    experiment_builder.setup_storage(local_config)

    storage = get_storage()

    upgrade_db_specifics(storage)

    print('Updating documents...')
    upgrade_documents(storage)
    print('Database upgrade completed successfully')


def upgrade_db_specifics(storage):
    """Make upgrades that are specific to some backends"""
    if isinstance(storage, Legacy):
        database = storage._db  # pylint: disable=protected-access
        print('Updating indexes...')
        update_indexes(database)
        if isinstance(database, PickledDB):
            print('Updating pickledb scheme...')
            upgrade_pickledb(database)
        elif isinstance(database, MongoDB):
            print('Updating mongodb scheme...')
            upgrade_mongodb(database)


def upgrade_documents(storage):
    """Upgrade scheme of the documents"""
    for experiment in storage.fetch_experiments({}):
        add_version(experiment)
        add_priors(experiment)
        storage.update_experiment(uid=experiment.pop('_id'), **experiment)


def add_version(experiment):
    """Add version 1 if not present"""
    experiment.setdefault('version', 1)


def add_priors(experiment):
    """Add priors to metadata if not present"""
    backward.populate_priors(experiment['metadata'])


def update_indexes(database):
    """Remove user from unique indices.

    This is required for migration to v0.1.6+
    """
    # For backward compatibility
    index_info = database.index_information('experiments')
    deprecated_indices = [('name', 'metadata.user'), ('name', 'metadata.user', 'version'),
                          'name_1_metadata.user_1', 'name_1_metadata.user_1_version_1']

    for deprecated_idx in deprecated_indices:
        if deprecated_idx in index_info:
            database.drop_index('experiments', deprecated_idx)


# pylint: disable=unused-argument
def upgrade_mongodb(database):
    """Update mongo specific db scheme."""
    pass


def upgrade_pickledb(database):
    """Update pickledb specific db scheme."""
    # pylint: disable=protected-access
    def upgrade_state(self, state):
        """Set state while ensuring backward compatibility"""
        self._documents = state['_documents']

        # if indexes are from <=v0.1.6
        if state['_indexes'] and isinstance(next(iter(state['_indexes'].keys())), tuple):
            self._indexes = dict()
            for keys, values in state['_indexes'].items():
                if isinstance(keys, str):
                    self._indexes[keys] = values
                # Convert keys that were registered with old index signature
                else:
                    keys = [(key, None) for key in keys]
                    self.create_index(keys, unique=True)
        else:
            self._indexes = state['_indexes']

    old_setstate = getattr(EphemeralCollection, '__setstate__', None)
    EphemeralCollection.__setstate__ = upgrade_state

    document = database.read('experiments', {})[0]
    # One document update is enough to fix all collections
    database.write('experiments', document, query={'_id': document['_id']})

    if old_setstate is not None:
        EphemeralCollection.__setstate__ = old_setstate
    else:
        del EphemeralCollection.__setstate__
