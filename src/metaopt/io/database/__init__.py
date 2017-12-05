# -*- coding: utf-8 -*-
"""
:mod:`metaopt.io.database` -- Wrappers for database frameworks
==============================================================

.. module:: database
   :platform: Unix
   :synopsis: Import name for wrappers of database frameworks.

Contains :class:`AbstractDB`, an interface for databases.
Currently, implemented wrappers:

   - :class:`metaopt.io.database.mongodb.MongoDB`

"""
from __future__ import absolute_import

from abc import abstractmethod, abstractproperty
import logging

import six

from metaopt.utils import (AbstractSingletonType, SingletonFactory)


@six.add_metaclass(AbstractSingletonType)
class AbstractDB(object):
    """Base class for database framework wrappers.

    Attributes
    ----------
    host : str
       It can be either:
          1. Known hostname or IP address in which database server resides.
          2. URI: A database framework specific connection string.
    dbname : str
       Name of database containing experiments.
    port : int
       Port that database server listens to for requests.
    username : str
        Name of user with write/read permissions to database with name `dbname`.
    password : str
        Secret phrase of user, `username`.

    """

    def __init__(self, host='localhost', dbname=None,
                 port=None, username=None, password=None):
        """Init method, see attributes of :class:`AbstractDB`."""
        self._logger = logging.getLogger(__name__)

        self.host = host
        self.dbname = dbname
        self.port = port
        self.username = username
        self.password = password

        self._db = None
        self._conn = None
        self.initiate_connection()

    @abstractproperty
    def is_connected(self):
        """True, if practical connection has been achieved."""
        pass

    @abstractmethod
    def initiate_connection(self):
        """Connect to database, unless `AbstractDB` `is_connected`.

        :raises :exc:`DatabaseError`: if connection or authentication fails

        """
        pass

    @abstractmethod
    def close_connection(self):
        """Disconnect from database, if `AbstractDB` `is_connected`."""
        pass

    @abstractmethod
    def write(self, collection_name, data,
              query=None):
        """Write new information to a collection. Perform insert or update.

        Parameters
        ----------
        collection_name : str
           A collection inside database, a table.
        data : dict or list of dicts
           New data that will **be inserted** or that will **update** entries.
        query : dict, optional
           Assumes an update operation: filter entries in collection to be updated.

        :return: operation success.

        .. note::
           In the case of an insert operation, `data` variable will be updated
           to contain a unique *_id* key.

        .. note::
           In the case of an update operation, if `query` fails to find a
           document that matches, insert of `data` will be performed instead.

        """
        pass

    @abstractmethod
    def read(self, collection_name, query=None, selection=None):
        """Read a collection and return a value according to the query.

        Parameters
        ----------
        collection_name : str
           A collection inside database, a table.
        query : dict, optional
           Filter entries in collection.
        selection : dict, optional
           Elements of matched entries to return, the projection.

        :return: list of matched document[s]

        """
        pass

    @abstractmethod
    def remove(self, collection_name, query):
        """Delete from a collection document[s] which match the `query`.

        Parameters
        ----------
        collection_name : str
           A collection inside database, a table.
        query : dict
           Filter entries in collection.

        :return: operation success.

        """
        pass


class DatabaseError(RuntimeError):
    """Exception type used to delegate responsibility from any database
    implementation's own Exception types.
    """

    pass


@six.add_metaclass(SingletonFactory)  # pylint: disable=too-few-public-methods,abstract-method
class Database(AbstractDB):
    """Class used to inject dependency on a database framework.

    .. seealso:: `Factory` metaclass and `AbstractDB` interface.
    """

    pass
