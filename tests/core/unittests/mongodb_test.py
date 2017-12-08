#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`metaopt.io.database.mongodb`."""

from datetime import datetime

import pytest

from metaopt.io.database import DatabaseError
from metaopt.io.database.mongodb import MongoDB


@pytest.fixture(scope='module')
def moptdb():
    """Return MongoDB wrapper instance initiated with test opts."""
    MongoDB.instance = None
    moptdb = MongoDB(username='user', password='pass', name='metaopt_test')
    return moptdb


@pytest.mark.usefixtures("null_db_instances")
class TestConnection(object):
    """Create a :class:`metaopt.io.database.mongodb.MongoDB`, check connection cases."""

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
class TestRead(object):
    """Calls to :meth:`metaopt.io.database.mongodb.MongoDB.read`."""

    def test_read_experiment(self, exp_config, moptdb):
        """Fetch a whole experiment entries."""
        loaded_config = moptdb.read(
            'experiments', {'name': 'supernaedo2', 'metadata.user': 'tsirif'})
        assert loaded_config == exp_config[0][:2]

        loaded_config = moptdb.read('experiments',
                                    {'name': 'supernaedo2',
                                     'metadata.user': 'tsirif',
                                     'metadata.datetime': datetime(2017, 11, 22, 23, 0, 0)})
        assert loaded_config == [exp_config[0][0]]
        assert loaded_config[0]['_id'] == exp_config[0][0]['_id']

    def test_read_with_id(self, exp_config, moptdb):
        """Query using ``_id`` key."""
        loaded_config = moptdb.read('experiments', {'_id': exp_config[0][2]['_id']})
        assert loaded_config == [exp_config[0][2]]

    def test_read_default(self, exp_config, moptdb):
        """Fetch value(s) from an entry."""
        value = moptdb.read(
            'experiments', {'name': 'supernaedo2', 'metadata.user': 'tsirif'},
            selection={'algorithms': 1, '_id': 0})
        assert value == [{'algorithms': exp_config[0][i]['algorithms']} for i in (0, 1)]

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
        assert value == exp_config[1][1:]

        value = moptdb.read(
            'trials',
            {'experiment': 'supernaedo2',
             'submit_time': {'$gt': datetime(2017, 11, 23, 0, 0, 0)}})
        assert value == exp_config[1][2:]


@pytest.mark.usefixtures("clean_db")
class TestWrite(object):
    """Calls to :meth:`metaopt.io.database.mongodb.MongoDB.write`."""

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
        filt = {'name': 'supernaedo2', 'metadata.user': 'tsirif'}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.write('experiments', {'pool_size': 16}, filt) is True
        assert database.experiments.count() == count_before
        value = list(database.experiments.find({'name': 'supernaedo2'}))
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
class TestRemove(object):
    """Calls to :meth:`metaopt.io.database.mongodb.MongoDB.remove`."""

    def test_remove_many_default(self, exp_config, database, moptdb):
        """Should match existing entries, and delete them all."""
        filt = {'name': 'supernaedo2', 'metadata.user': 'tsirif'}
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
    """Calls :meth:`metaopt.io.database.mongodb.MongoDB.count`."""

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
