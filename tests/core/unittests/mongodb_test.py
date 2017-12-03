#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`metaopt.io.database.mongodb`."""

from datetime import datetime
import os

from pymongo import MongoClient
import pytest
import yaml

from metaopt.io.database import (DatabaseError, MongoDB)

DB_TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def exp_config():
    """Load an example database."""
    with open(os.path.join(DB_TEST_DIR, 'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))
    return exp_config


@pytest.fixture(scope='module')
def database():
    """Return Mongo database object to test with example entries."""
    client = MongoClient(username='user', password='pass', authSource='metaopt_test')
    database = client.metaopt_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database, exp_config):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.experiments.insert_many(exp_config[0])
    database.trials.drop()
    database.trials.insert_many(exp_config[1])
    database.workers.drop()
    database.workers.insert_many(exp_config[2])
    database.resources.drop()
    database.resources.insert_many(exp_config[3])


@pytest.fixture(scope='module')
def moptdb():
    """Return MongoDB wrapper instance initiated with test opts."""
    moptdb = MongoDB(username='user', password='pass', dbname='metaopt_test')
    return moptdb


class TestConnection(object):
    """Create a :class:`metaopt.io.database.MongoDB`, check connection cases."""

    def test_bad_connection(self):
        """Raise when connection cannot be achieved."""
        with pytest.raises(DatabaseError) as exc_info:
            MongoDB(host='asdfada', port=123, dbname='metaopt',
                    username='uasdfaf', password='paasdfss')
        assert "Connection" in str(exc_info.value)

    def test_bad_authentication(self):
        """Raise when authentication cannot be achieved."""
        with pytest.raises(DatabaseError) as exc_info:
            MongoDB(dbname='metaopt_test', username='uasdfaf', password='paasdfss')
        assert "Authentication" in str(exc_info.value)

    def test_connection_with_uri(self):
        """Check the case when connecting with ready `uri`."""
        moptdb = MongoDB('mongodb://user:pass@localhost/metaopt_test')
        assert moptdb.host == 'localhost'
        assert moptdb.port == 27017
        assert moptdb.username == 'user'
        assert moptdb.password == 'pass'
        assert moptdb.dbname == 'metaopt_test'

    def test_overwrite_uri(self):
        """Check the case when connecting with ready `uri`."""
        moptdb = MongoDB('mongodb://lala:pass@localhost:1231/metaopt',
                         port=27017, dbname='metaopt_test', username='user',
                         password='pass')
        assert moptdb.host == 'localhost'
        assert moptdb.port == 27017
        assert moptdb.username == 'user'
        assert moptdb.password == 'pass'
        assert moptdb.dbname == 'metaopt_test'

    def test_singleton(self):
        """Test that MongoDB class is a singleton."""
        moptdb = MongoDB(username='user', password='pass', dbname='metaopt_test')
        # reinit connection does not change anything
        moptdb.initiate_connection()
        moptdb.close_connection()
        assert MongoDB() is moptdb


@pytest.mark.usefixtures("clean_db")
class TestRead(object):
    """Calls to :meth:`metaopt.io.database.MongoDB.read`."""

    def test_read_experiment(self, exp_config, moptdb):
        """Fetch a whole experiment entries."""
        loaded_config = moptdb.read(
            'experiments', {'exp_name': 'supernaedo2', 'metadata.user': 'tsirif'})
        assert loaded_config == exp_config[0][:2]

        loaded_config = moptdb.read('experiments',
                                    {'exp_name': 'supernaedo2',
                                     'metadata.user': 'tsirif',
                                     'metadata.datetime': datetime(2017, 11, 22, 23, 0, 0)})
        assert loaded_config == [exp_config[0][0]]

    def test_read_default(self, exp_config, moptdb):
        """Fetch value(s) from an entry."""
        value = moptdb.read(
            'experiments', {'exp_name': 'supernaedo2', 'metadata.user': 'tsirif'},
            selection={'algorithms': 1, '_id': 0})
        assert value == [{'algorithms': exp_config[0][i]['algorithms']} for i in (0, 1)]

    def test_read_nothing(self, moptdb):
        """Fetch value(s) from an entry."""
        value = moptdb.read(
            'experiments', {'exp_name': 'not_found', 'metadata.user': 'tsirif'},
            selection={'algorithms': 1})
        assert value == []

    def test_read_trials(self, exp_config, moptdb):
        """Fetch value(s) from an entry."""
        value = moptdb.read(
            'trials',
            {'exp_name': 'supernaedo2', 'user': 'tsirif',
             'submit_time': {'$gte': datetime(2017, 11, 23, 0, 0, 0)}})
        assert value == exp_config[1][1:]

        value = moptdb.read(
            'trials',
            {'exp_name': 'supernaedo2', 'user': 'tsirif',
             'submit_time': {'$gt': datetime(2017, 11, 23, 0, 0, 0)}})
        assert value == exp_config[1][2:]


class TestWrite(object):
    """Calls to :meth:`metaopt.io.database.MongoDB.write`."""

    def test_insert_one(self, database, moptdb):
        """Should insert a single new entry in the collection."""
        item = {'exp_name': 'supernaekei',
                'metadata': {'user': 'tsirif'}}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.write('experiments', item) is True
        assert database.experiments.count() == count_before + 1
        value = database.experiments.find_one({'exp_name': 'supernaekei'})
        assert value == item

    def test_insert_many(self, database, moptdb):
        """Should insert two new entry (as a list) in the collection."""
        item = [{'exp_name': 'supernaekei2',
                 'metadata': {'user': 'tsirif'}},
                {'exp_name': 'supernaekei3',
                 'metadata': {'user': 'tsirif'}}]
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
        filt = {'exp_name': 'supernaedo2', 'metadata.user': 'tsirif'}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.write('experiments', {'pool_size': 16}, filt) is True
        assert database.experiments.count() == count_before
        value = list(database.experiments.find({'exp_name': 'supernaedo2'}))
        assert value[0]['pool_size'] == 16
        assert value[1]['pool_size'] == 16
        assert value[2]['pool_size'] == 2


@pytest.mark.usefixtures("clean_db")
class TestRemove(object):
    """Calls to :meth:`metaopt.io.database.MongoDB.remove`."""

    def test_remove_many_default(self, database, moptdb):
        """Should match existing entries, and delete them all."""
        filt = {'exp_name': 'supernaedo2', 'metadata.user': 'tsirif'}
        count_before = database.experiments.count()
        # call interface
        assert moptdb.remove('experiments', filt) is True
        assert database.experiments.count() == count_before - 2
