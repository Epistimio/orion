#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`metaopt.core.io.database.mongodb`."""

from datetime import datetime
import functools

import pymongo
from pymongo import MongoClient
import pytest

from metaopt.core.io.database import Database, DatabaseError, DuplicateKeyError
from metaopt.core.io.database.mongodb import MongoDB


@pytest.fixture(scope='module')
def moptdb():
    """Return MongoDB wrapper instance initiated with test opts."""
    MongoDB.instance = None
    moptdb = MongoDB(username='user', password='pass', name='metaopt_test')
    return moptdb


@pytest.fixture()
def patch_mongo_client(monkeypatch):
    """Patch ``pymongo.MongoClient`` to force serverSelectionTimeoutMS to 1."""
    def mock_class(*args, **kwargs):
        # 1 sec, defaults to 20 secs otherwise
        kwargs['serverSelectionTimeoutMS'] = 1.
        # NOTE: Can't use pymongo.MongoClient otherwise there is an infinit
        # recursion; mock(mock(mock(mock(...(MongoClient)...))))
        return MongoClient(*args, **kwargs)

    monkeypatch.setattr('pymongo.MongoClient', mock_class)


@pytest.mark.usefixtures("null_db_instances")
class TestConnection(object):
    """Create a :class:`metaopt.core.io.database.mongodb.MongoDB`, check connection cases."""

    @pytest.mark.usefixtures("patch_mongo_client")
    def test_bad_connection(self):
        """Raise when connection cannot be achieved."""
        with pytest.raises(DatabaseError) as exc_info:
            MongoDB(host='asdfada', port=123, name='metaopt',
                    username='uasdfaf', password='paasdfss')
        assert "Connection" in str(exc_info.value)

    def test_bad_authentication(self):
        """Raise when authentication cannot be achieved."""
        with pytest.raises(DatabaseError) as exc_info:
            MongoDB(name='metaopt_test', username='uasdfaf', password='paasdfss')
        assert "Authentication" in str(exc_info.value)

    def test_connection_with_uri(self):
        """Check the case when connecting with ready `uri`."""
        moptdb = MongoDB('mongodb://user:pass@localhost/metaopt_test')
        assert moptdb.host == 'localhost'
        assert moptdb.port == 27017
        assert moptdb.username == 'user'
        assert moptdb.password == 'pass'
        assert moptdb.name == 'metaopt_test'

    def test_overwrite_uri(self):
        """Check the case when connecting with ready `uri`."""
        moptdb = MongoDB('mongodb://lala:pass@localhost:1231/metaopt',
                         port=27017, name='metaopt_test', username='user',
                         password='pass')
        assert moptdb.host == 'localhost'
        assert moptdb.port == 27017
        assert moptdb.username == 'user'
        assert moptdb.password == 'pass'
        assert moptdb.name == 'metaopt_test'

    def test_singleton(self):
        """Test that MongoDB class is a singleton."""
        moptdb = MongoDB(username='user', password='pass', name='metaopt_test')
        # reinit connection does not change anything
        moptdb.initiate_connection()
        moptdb.close_connection()
        assert MongoDB() is moptdb


@pytest.mark.usefixtures("clean_db")
class TestExceptionWrapper(object):
    """Call to methods wrapped with `mongodb_exception_wrapper()`."""

    def test_duplicate_key_error(self, monkeypatch, moptdb, exp_config):
        """Should raise generic DuplicateKeyError."""
        # Add unique indexes to force trigger of DuplicateKeyError on write()
        moptdb.ensure_index('experiments',
                            [('name', Database.ASCENDING),
                             ('metadata.user', Database.ASCENDING)],
                            unique=True)

        config_to_add = exp_config[0][0]
        config_to_add.pop('_id')

        query = {'_id': exp_config[0][1]['_id']}

        # Make sure it raises pymongo.errors.DuplicateKeyError when there is no
        # wrapper
        monkeypatch.setattr(
            moptdb, "read_and_write",
            functools.partial(moptdb.read_and_write.__wrapped__, moptdb))
        with pytest.raises(pymongo.errors.DuplicateKeyError) as exc_info:
            moptdb.read_and_write('experiments', query, config_to_add)

        monkeypatch.undo()

        # Verify that the wrapper converts it properly to DuplicateKeyError
        with pytest.raises(DuplicateKeyError) as exc_info:
            moptdb.read_and_write('experiments', query, config_to_add)
        assert "duplicate key error" in str(exc_info.value)

    def test_bulk_duplicate_key_error(self, monkeypatch, moptdb, exp_config):
        """Should raise generic DuplicateKeyError."""
        # Make sure it raises pymongo.errors.BulkWriteError when there is no
        # wrapper
        monkeypatch.setattr(
            moptdb, "write",
            functools.partial(moptdb.write.__wrapped__, moptdb))
        with pytest.raises(pymongo.errors.BulkWriteError) as exc_info:
            moptdb.write('experiments', exp_config[0])

        monkeypatch.undo()

        # Verify that the wrapper converts it properly to DuplicateKeyError
        with pytest.raises(DuplicateKeyError) as exc_info:
            moptdb.write('experiments', exp_config[0])
        assert "duplicate key error" in str(exc_info.value)

    def test_non_converted_errors(self, moptdb, exp_config):
        """Should raise OperationFailure.

        This is because _id inside exp_config[0][0] cannot be set. It is an
        immutable key of the collection.

        """
        config_to_add = exp_config[0][0]

        query = {'_id': exp_config[0][1]['_id']}

        with pytest.raises(pymongo.errors.OperationFailure):
            moptdb.read_and_write('experiments', query, config_to_add)


@pytest.mark.usefixtures("clean_db")
class TestEnsureIndex(object):
    """Calls to :meth:`metaopt.core.io.database.mongodb.MongoDB.ensure_index`."""

    def test_new_index(self, moptdb):
        """Index should be added to mongo database"""
        assert "status_1" not in moptdb._db.experiments.index_information()
        moptdb.ensure_index('experiments', 'status')
        assert "status_1" in moptdb._db.experiments.index_information()

    def test_existing_index(self, moptdb):
        """Index should be added to mongo database and reattempt should do nothing"""
        assert "status_1" not in moptdb._db.experiments.index_information()
        moptdb.ensure_index('experiments', 'status')
        assert "status_1" in moptdb._db.experiments.index_information()
        moptdb.ensure_index('experiments', 'status')
        assert "status_1" in moptdb._db.experiments.index_information()

    def test_ordered_index(self, moptdb):
        """Sort order should be added to index"""
        assert "end_time_-1" not in moptdb._db.trials.index_information()
        moptdb.ensure_index('trials', [('end_time', MongoDB.DESCENDING)])
        assert "end_time_-1" in moptdb._db.trials.index_information()

    def test_compound_index(self, moptdb):
        """Tuple of Index should be added as a compound index."""
        assert "name_1_metadata.user_1" not in moptdb._db.experiments.index_information()
        moptdb.ensure_index('experiments',
                            [('name', MongoDB.ASCENDING),
                             ('metadata.user', MongoDB.ASCENDING)])
        assert "name_1_metadata.user_1" in moptdb._db.experiments.index_information()

    def test_unique_index(self, moptdb):
        """Index should be set as unique in mongo database's index information."""
        assert "name_1_metadata.user_1" not in moptdb._db.experiments.index_information()
        moptdb.ensure_index('experiments',
                            [('name', MongoDB.ASCENDING),
                             ('metadata.user', MongoDB.ASCENDING)],
                            unique=True)
        index_information = moptdb._db.experiments.index_information()
        assert "name_1_metadata.user_1" in index_information
        assert index_information["name_1_metadata.user_1"]['unique']


@pytest.mark.usefixtures("clean_db")
class TestRead(object):
    """Calls to :meth:`metaopt.core.io.database.mongodb.MongoDB.read`."""

    def test_read_experiment(self, exp_config, moptdb):
        """Fetch a whole experiment entries."""
        loaded_config = moptdb.read(
            'trials', {'experiment': 'supernaedo2', 'status': 'new'})
        assert loaded_config == [exp_config[1][3], exp_config[1][4]]

        loaded_config = moptdb.read(
            'trials',
            {'experiment': 'supernaedo2',
             'submit_time': exp_config[1][3]['submit_time']})
        assert loaded_config == [exp_config[1][3]]
        assert loaded_config[0]['_id'] == exp_config[1][3]['_id']

    def test_read_with_id(self, exp_config, moptdb):
        """Query using ``_id`` key."""
        loaded_config = moptdb.read('experiments', {'_id': exp_config[0][2]['_id']})
        assert loaded_config == [exp_config[0][2]]

    def test_read_default(self, exp_config, moptdb):
        """Fetch value(s) from an entry."""
        value = moptdb.read(
            'experiments', {'name': 'supernaedo2', 'metadata.user': 'tsirif'},
            selection={'algorithms': 1, '_id': 0})
        assert value == [{'algorithms': exp_config[0][0]['algorithms']}]

    def test_read_nothing(self, moptdb):
        """Fetch value(s) from an entry."""
        value = moptdb.read(
            'experiments', {'name': 'not_found', 'metadata.user': 'tsirif'},
            selection={'algorithms': 1})
        assert value == []

    def test_read_trials(self, exp_config, moptdb):
        """Fetch value(s) from an entry."""
        value = moptdb.read(
            'trials',
            {'experiment': 'supernaedo2',
             'submit_time': {'$gte': datetime(2017, 11, 23, 0, 0, 0)}})
        assert value == exp_config[1][2:]

        value = moptdb.read(
            'trials',
            {'experiment': 'supernaedo2',
             'submit_time': {'$gt': datetime(2017, 11, 23, 0, 0, 0)}})
        assert value == exp_config[1][3:]


@pytest.mark.usefixtures("clean_db")
class TestWrite(object):
    """Calls to :meth:`metaopt.core.io.database.mongodb.MongoDB.write`."""

    def test_insert_one(self, database, moptdb):
        """Should insert a single new entry in the collection."""
        item = {'exp_name': 'supernaekei',
                'user': 'tsirif'}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.write('experiments', item) is True
        assert database.experiments.count() == count_before + 1
        value = database.experiments.find_one({'exp_name': 'supernaekei'})
        assert value == item

    def test_insert_many(self, database, moptdb):
        """Should insert two new entry (as a list) in the collection."""
        item = [{'exp_name': 'supernaekei2',
                 'user': 'tsirif'},
                {'exp_name': 'supernaekei3',
                 'user': 'tsirif'}]
        count_before = database.experiments.count()
        # call interface
        assert moptdb.write('experiments', item) is True
        assert database.experiments.count() == count_before + 2
        value = database.experiments.find_one({'exp_name': 'supernaekei2'})
        assert value == item[0]
        value = database.experiments.find_one({'exp_name': 'supernaekei3'})
        assert value == item[1]

    def test_update_many_default(self, database, moptdb):
        """Should match existing entries, and update some of their keys."""
        filt = {'metadata.user': 'tsirif'}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.write('experiments', {'pool_size': 16}, filt) is True
        assert database.experiments.count() == count_before
        value = list(database.experiments.find({}))
        assert value[0]['pool_size'] == 16
        assert value[1]['pool_size'] == 16
        assert value[2]['pool_size'] == 2

    def test_update_with_id(self, exp_config, database, moptdb):
        """Query using ``_id`` key."""
        filt = {'_id': exp_config[0][1]['_id']}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.write('experiments', {'pool_size': 36}, filt) is True
        assert database.experiments.count() == count_before
        value = list(database.experiments.find())
        assert value[0]['pool_size'] == 2
        assert value[1]['pool_size'] == 36
        assert value[2]['pool_size'] == 2

    def test_upsert_with_id(self, database, moptdb):
        """Query with a non-existent ``_id`` should upsert something."""
        filt = {'_id': 'lalalathisisnew'}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.write('experiments', {'pool_size': 66}, filt) is True
        assert database.experiments.count() == count_before + 1
        value = list(database.experiments.find(filt))
        assert len(value) == 1
        assert len(value[0]) == 2
        assert value[0]['_id'] == 'lalalathisisnew'
        assert value[0]['pool_size'] == 66


@pytest.mark.usefixtures("clean_db")
class TestReadAndWrite(object):
    """Calls to :meth:`metaopt.core.io.database.mongodb.MongoDB.read_and_write`."""

    def test_read_and_write_one(self, database, moptdb, exp_config):
        """Should read and update a single entry in the collection."""
        # Make sure there is only one match
        documents = moptdb.read(
            'experiments',
            {'name': 'supernaedo2', 'metadata.user': 'dendi'})
        assert len(documents) == 1

        # Find and update atomically
        loaded_config = moptdb.read_and_write(
            'experiments',
            {'name': 'supernaedo2', 'metadata.user': 'dendi'},
            {'status': 'lalala'})
        exp_config[0][2]['status'] = 'lalala'
        assert loaded_config == exp_config[0][2]

    def test_read_and_write_many(self, database, moptdb, exp_config):
        """Should update only one entry."""
        # Make sure there is many matches
        documents = moptdb.read('experiments', {'name': 'supernaedo2'})
        assert len(documents) > 1

        # Find many and update first one only
        loaded_config = moptdb.read_and_write(
            'experiments',
            {'name': 'supernaedo2'},
            {'status': 'lalala'})

        exp_config[0][0]['status'] = 'lalala'
        assert loaded_config == exp_config[0][0]

        # Make sure it only changed the first document found
        documents = moptdb.read('experiments', {'name': 'supernaedo2'})
        assert documents[0]['status'] == 'lalala'
        assert documents[1]['status'] != 'lalala'

    def test_read_and_write_no_match(self, database, moptdb):
        """Should return None when there is no match."""
        loaded_config = moptdb.read_and_write(
            'experiments',
            {'name': 'lalala'},
            {'status': 'lalala'})

        assert loaded_config is None


@pytest.mark.usefixtures("clean_db")
class TestRemove(object):
    """Calls to :meth:`metaopt.core.io.database.mongodb.MongoDB.remove`."""

    def test_remove_many_default(self, exp_config, database, moptdb):
        """Should match existing entries, and delete them all."""
        filt = {'metadata.user': 'tsirif'}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.remove('experiments', filt) is True
        assert database.experiments.count() == count_before - 2
        assert database.experiments.count() == 1
        assert list(database.experiments.find()) == [exp_config[0][2]]

    def test_remove_with_id(self, exp_config, database, moptdb):
        """Query using ``_id`` key."""
        filt = {'_id': exp_config[0][0]['_id']}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.remove('experiments', filt) is True
        assert database.experiments.count() == count_before - 1
        assert list(database.experiments.find()) == exp_config[0][1:]


@pytest.mark.usefixtures("clean_db")
class TestCount(object):
    """Calls :meth:`metaopt.core.io.database.mongodb.MongoDB.count`."""

    def test_count_default(self, exp_config, moptdb):
        """Call just with collection name."""
        found = moptdb.count('trials')
        assert found == len(exp_config[1])

    def test_count_query(self, exp_config, moptdb):
        """Call with a query."""
        found = moptdb.count('trials', {'status': 'completed'})
        assert found == len([x for x in exp_config[1] if x['status'] == 'completed'])

    def test_count_query_with_id(self, exp_config, moptdb):
        """Call querying with unique _id."""
        found = moptdb.count('trials', {'_id': exp_config[1][2]['_id']})
        assert found == 1

    def test_count_nothing(self, moptdb):
        """Call with argument that will not find anything."""
        found = moptdb.count('experiments', {'name': 'lalalanotfound'})
        assert found == 0
