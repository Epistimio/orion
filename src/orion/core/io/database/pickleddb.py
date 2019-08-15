# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.database.pickleddb` -- Pickled Database
===========================================================

.. module:: database
   :platform: Unix
   :synopsis: Implement permanent version of :class:`orion.core.io.database.EphemeralDB`

"""

from contextlib import contextmanager
import logging
import os
import pickle
from pickle import PicklingError

from filelock import FileLock

import orion.core
from orion.core.io.database import AbstractDB
from orion.core.io.database.ephemeraldb import EphemeralDB

log = logging.getLogger(__name__)

DEFAULT_HOST = os.path.join(orion.core.DIRS.user_data_dir, 'orion', 'orion_db.pkl')


def find_unpickable_doc(dict_of_dict):
    for name, collection in dict_of_dict:
        documents = collection.find()

        for doc in documents:
            try:
                pickle.dumps(doc)

            except PicklingError:
                return name, doc

    return None, None


def find_unpickable_field(doc):
    for k, v in doc.to_dict().items():
        try:
            pickle.dumps(v)

        except PicklingError:
            return k, v

    return None, None


class PickledDB(AbstractDB):
    """Pickled EphemeralDB to support permanancy and concurrency

    This is a very simple and inefficient implementation of a permanent database on disk for Oríon.
    The data is loaded from disk for every operation, and every operation is protected with a
    filelock.

    Parameters
    ----------
    host: str
        File path to save pickled ephemeraldb.  Default is {user data dir}/orion/orion_db.pkl ex:
        $HOME/.local/share/orion/orion_db.pkl

    """

    # pylint: disable=unused-argument
    def __init__(self, host=DEFAULT_HOST, *args, **kwargs):
        super(PickledDB, self).__init__(host)

        if os.path.dirname(host):
            os.makedirs(os.path.dirname(host), exist_ok=True)

    @property
    def is_connected(self):
        """Return true, always."""
        return True

    def initiate_connection(self):
        """Do nothing"""
        pass

    def close_connection(self):
        """Do nothing"""
        pass

    def ensure_index(self, collection_name, keys, unique=False):
        """Create given indexes if they do not already exist in database.

        Indexes are only created if `unique` is True.
        """
        with self.locked_database() as database:
            database.ensure_index(collection_name, keys, unique=unique)

    def write(self, collection_name, data, query=None):
        """Write new information to a collection. Perform insert or update.

        .. seealso:: :meth:`AbstractDB.write` for argument documentation.

        """
        with self.locked_database() as database:
            return database.write(collection_name, data, query=query)

    def read(self, collection_name, query=None, selection=None):
        """Read a collection and return a value according to the query.

        .. seealso:: :meth:`AbstractDB.read` for argument documentation.

        """
        with self.locked_database(write=False) as database:
            return database.read(collection_name, query=query, selection=selection)

    def read_and_write(self, collection_name, query, data, selection=None):
        """Read a collection's document and update the found document.

        Returns the updated document, or None if nothing found.

        .. seealso:: :meth:`AbstractDB.read_and_write` for
                     argument documentation.

        """
        with self.locked_database() as database:
            return database.read_and_write(collection_name, query=query, data=data,
                                           selection=selection)

    def count(self, collection_name, query=None):
        """Count the number of documents in a collection which match the `query`.

        .. seealso:: :meth:`AbstractDB.count` for argument documentation.

        """
        with self.locked_database(write=False) as database:
            return database.count(collection_name, query=query)

    def remove(self, collection_name, query):
        """Delete from a collection document[s] which match the `query`.

        .. seealso:: :meth:`AbstractDB.remove` for argument documentation.

        """
        with self.locked_database() as database:
            return database.remove(collection_name, query=query)

    def _get_database(self):
        """Read fresh DB state from pickled file"""
        if not os.path.exists(self.host):
            return EphemeralDB()

        with open(self.host, 'rb') as f:
            database = pickle.load(f)

        return database

    # pylint: disable: protected-access
    def _dump_database(self, database):
        """Write pickled DB on disk"""
        tmp_file = self.host + '.tmp'

        try:
            with open(tmp_file, 'wb') as f:
                pickle.dump(database, f)

        except PicklingError:
            collection, doc = find_unpickable_doc(database._db)
            log.error('Document in (collection: %s) is not pickable\ndoc: %s',
                      collection, doc._data)

            key, value = find_unpickable_field(doc)
            log.error('because (value %s) in (field: %s) is not pickable',
                      value, key)
            raise

        os.rename(tmp_file, self.host)

    @contextmanager
    def locked_database(self, write=True):
        """Lock database file during wrapped operation call."""
        lock = FileLock(self.host + '.lock')

        with lock.acquire(timeout=60):
            database = self._get_database()

            yield database

            if write:
                self._dump_database(database)
