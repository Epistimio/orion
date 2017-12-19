# -*- coding: utf-8 -*-
"""
:mod:`metaopt.io.database.mongodb` -- Wrapper for MongoDB
=========================================================

.. module:: database
   :platform: Unix
   :synopsis: Implement :class:`metaopt.io.database.AbstractDB` for MongoDB.

"""
from __future__ import absolute_import

import pymongo
from pymongo import MongoClient
from pymongo.uri_parser import parse_uri
from six import raise_from

from metaopt.io.database import (AbstractDB, DatabaseError)


class MongoDB(AbstractDB):
    """Wrap MongoDB with three primary methods `read`, `write`, `remove`.

    Attributes
    ----------
    host : str
       Hostname or MongoDB compliant full credentials+address+database
       specification.

    Information on MongoDB `connection string
    <https://docs.mongodb.com/manual/reference/connection-string/>`_.

    .. seealso:: :class:`metaopt.io.database.AbstractDB` for more on attributes.

    """

    def initiate_connection(self):
        """Connect to database, unless MongoDB `is_connected`.

        :raises :exc:`DatabaseError`: if connection or authentication fails

        """
        if self.is_connected:
            return

        self._sanitize_attrs()

        try:
            self._conn = MongoClient(host=self.host,
                                     port=self.port,
                                     username=self.username,
                                     password=self.password,
                                     authSource=self.name)
            self._db = self._conn[self.name]
            self._db.command('ismaster')  # .. seealso:: :meth:`is_connected`
        except pymongo.errors.ConnectionFailure as e:
            self._logger.error("Could not connect to host, %s:%s",
                               self.host, self.port)
            raise_from(DatabaseError("Connection Failure: database not found on "
                                     "specified uri"), e)
        except pymongo.errors.OperationFailure as e:
            self._logger.error("Could not verify user, %s, on database, %s",
                               self.username, self.name)
            raise_from(DatabaseError("Authentication Failure: bad credentials"), e)

    @property
    def is_connected(self):
        """True, if practical connection has been achieved.

        .. note:: MongoDB does not do this automatically when creating the client.
        """
        try:
            self._db.command('ismaster')
        except (pymongo.errors.ConnectionFailure,
                pymongo.errors.OperationFailure,
                TypeError, AttributeError):
            _is_connected = False
        else:
            _is_connected = True
        return _is_connected

    def close_connection(self):
        """Disconnect from database.

        .. note:: Doesn't really do anything because MongoDB reopens connection,
           when a client or client-derived object is accessed.
        """
        self._conn.close()

    def write(self, collection_name, data,
              query=None):
        """Write new information to a collection. Perform insert or update.

        .. seealso:: :meth:`AbstractDB.write` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        if query is None:
            # We can assume that we do not want to update.
            # So we do insert_many instead.
            if type(data) not in (list, tuple):
                data = [data]
            result = dbcollection.insert_many(documents=data)
            return result.acknowledged

        update_data = {'$set': data}

        result = dbcollection.update_many(filter=query,
                                          update=update_data,
                                          upsert=True)
        return result.acknowledged

    def read(self, collection_name, query=None, selection=None):
        """Read a collection and return a value according to the query.

        .. seealso:: :meth:`AbstractDB.read` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        cursor = dbcollection.find(query, selection)
        dbdocs = list(cursor)

        return dbdocs

    def count(self, collection_name, query=None):
        """Count the number of documents in a collection which match the `query`.

        .. seealso:: :meth:`AbstractDB.count` for argument documentation.

        """
        dbcollection = self._db[collection_name]
        return dbcollection.count(filter=query)

    def remove(self, collection_name, query):
        """Delete from a collection document[s] which match the `query`.

        .. seealso:: :meth:`AbstractDB.remove` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        result = dbcollection.delete_many(filter=query)
        return result.acknowledged

    def _sanitize_attrs(self):
        """Sanitize attributes using MongoDB's 'uri_parser' module."""
        try:
            # Host can be a valid MongoDB URI
            settings = parse_uri(self.host)
        except pymongo.errors.InvalidURI:  # host argument was a hostname
            if self.port is None:
                self.port = MongoClient.PORT
        else:  # host argument was a URI
            # Arguments in MongoClient overwrite elements from URI
            self.host, _port = settings['nodelist'][0]
            if self.name is None:
                self.name = settings['database']
            if self.port is None:
                self.port = _port
            if self.username is None:
                self.username = settings['username']
            if self.password is None:
                self.password = settings['password']
