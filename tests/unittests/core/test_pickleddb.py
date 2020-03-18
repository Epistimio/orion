#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.io.database.pickleddb`."""
from datetime import datetime
import logging
from multiprocessing import Pool
import os

from filelock import FileLock, Timeout
import pytest

from orion.core.io.database import Database, DatabaseTimeout, DuplicateKeyError
from orion.core.io.database.ephemeraldb import EphemeralCollection
from orion.core.io.database.pickleddb import find_unpickable_doc, find_unpickable_field, PickledDB
import orion.core.utils.backward as backward


@pytest.fixture()
def orion_db():
    """Return PickledDB wrapper instance initiated with test opts."""
    PickledDB.instance = None
    orion_db = PickledDB(host='orion_db.pkl')
    yield orion_db


@pytest.fixture()
def clean_db(orion_db, exp_config):
    """Clean insert example experiment entries to collections."""
    if os.path.exists(orion_db.host):
        os.remove(orion_db.host)

    ephemeral_db = orion_db._get_database()
    database = ephemeral_db._db
    database['experiments'].drop()
    database['experiments'].insert_many(exp_config[0])
    database['trials'].drop()
    database['trials'].insert_many(exp_config[1])
    database['workers'].drop()
    database['workers'].insert_many(exp_config[2])
    database['resources'].drop()
    database['resources'].insert_many(exp_config[3])
    orion_db._dump_database(ephemeral_db)


@pytest.mark.usefixtures("clean_db")
class TestEnsureIndex(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.ensure_index`."""

    def test_new_index(self, orion_db):
        """Index should be added to pickled database"""
        assert "new_field_1" not in orion_db._get_database()._db['new_collection']._indexes

        orion_db.ensure_index('new_collection', 'new_field', unique=False)
        assert "new_field_1" not in orion_db._get_database()._db['new_collection']._indexes

        orion_db.ensure_index('new_collection', 'new_field', unique=True)
        assert "new_field_1" in orion_db._get_database()._db['new_collection']._indexes

    def test_existing_index(self, orion_db):
        """Index should be added to pickled database and reattempt should do nothing"""
        assert "new_field_1" not in orion_db._get_database()._db['new_collection']._indexes

        orion_db.ensure_index('new_collection', 'new_field', unique=True)
        assert "new_field_1" in orion_db._get_database()._db['new_collection']._indexes

        # reattempt
        orion_db.ensure_index('new_collection', 'new_field', unique=True)
        assert "new_field_1" in orion_db._get_database()._db['new_collection']._indexes

    def test_compound_index(self, orion_db):
        """Tuple of Index should be added as a compound index."""
        assert "name_1_metadata.user_1" not in orion_db._get_database()._db['experiments']._indexes
        orion_db.ensure_index('experiments',
                              [('name', Database.ASCENDING),
                               ('metadata.user', Database.ASCENDING)], unique=True)
        assert "name_1_metadata.user_1" in orion_db._get_database()._db['experiments']._indexes


@pytest.mark.usefixtures("clean_db")
class TestRead(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.read`."""

    def test_read_experiment(self, exp_config, orion_db):
        """Fetch a whole experiment entries."""
        loaded_config = orion_db.read(
            'trials', {'experiment': 'supernaedo2-dendi', 'status': 'new'})
        assert loaded_config == [exp_config[1][3], exp_config[1][4]]

        loaded_config = orion_db.read(
            'trials',
            {'experiment': 'supernaedo2-dendi',
             'submit_time': exp_config[1][3]['submit_time']})
        assert loaded_config == [exp_config[1][3]]
        assert loaded_config[0]['_id'] == exp_config[1][3]['_id']

    def test_read_with_id(self, exp_config, orion_db):
        """Query using ``_id`` key."""
        loaded_config = orion_db.read('experiments', {'_id': exp_config[0][2]['_id']})
        backward.populate_space(loaded_config[0])
        assert loaded_config == [exp_config[0][2]]

    def test_read_default(self, exp_config, orion_db):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            'experiments', {'name': 'supernaedo2', 'metadata.user': 'tsirif'},
            selection={'algorithms': 1, '_id': 0})
        assert value == [{'algorithms': exp_config[0][0]['algorithms']}]

    def test_read_nothing(self, orion_db):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            'experiments', {'name': 'not_found', 'metadata.user': 'tsirif'},
            selection={'algorithms': 1})
        assert value == []

    def test_read_trials(self, exp_config, orion_db):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            'trials',
            {'experiment': 'supernaedo2-dendi',
             'submit_time': {'$gte': datetime(2017, 11, 23, 0, 0, 0)}})
        assert value == [exp_config[1][1]] + exp_config[1][3:7]

        value = orion_db.read(
            'trials',
            {'experiment': 'supernaedo2-dendi',
             'submit_time': {'$gt': datetime(2017, 11, 23, 0, 0, 0)}})
        assert value == exp_config[1][3:7]

    def test_null_comp(self, exp_config, orion_db):
        """Fetch value(s) from an entry."""
        all_values = orion_db.read(
            'trials',
            {'experiment': 'supernaedo2-dendi',
             'end_time': {'$gte': datetime(2017, 11, 1, 0, 0, 0)}})

        db = orion_db._get_database()
        db._db['trials']._documents[0]._data['end_time'] = None
        orion_db._dump_database(db)

        values = orion_db.read(
            'trials',
            {'experiment': 'supernaedo2-dendi',
             'end_time': {'$gte': datetime(2017, 11, 1, 0, 0, 0)}})
        assert len(values) == len(all_values) - 1


@pytest.mark.usefixtures("clean_db")
class TestWrite(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.write`."""

    def test_insert_one(self, orion_db):
        """Should insert a single new entry in the collection."""
        item = {'exp_name': 'supernaekei',
                'user': 'tsirif'}
        count_before = orion_db._get_database().count('experiments')
        # call interface
        assert orion_db.write('experiments', item) == 1
        assert orion_db._get_database().count('experiments') == count_before + 1
        value = orion_db._get_database()._db['experiments'].find({'exp_name': 'supernaekei'})[0]
        assert value == item

    def test_insert_many(self, orion_db):
        """Should insert two new entry (as a list) in the collection."""
        item = [{'exp_name': 'supernaekei2',
                 'user': 'tsirif'},
                {'exp_name': 'supernaekei3',
                 'user': 'tsirif'}]
        count_before = orion_db._get_database()._db['experiments'].count()
        # call interface
        assert orion_db.write('experiments', item) == 2
        database = orion_db._get_database()._db
        assert database['experiments'].count() == count_before + 2
        value = database['experiments'].find({'exp_name': 'supernaekei2'})[0]
        assert value == item[0]
        value = database['experiments'].find({'exp_name': 'supernaekei3'})[0]
        assert value == item[1]

    def test_update_many_default(self, orion_db):
        """Should match existing entries, and update some of their keys."""
        filt = {'metadata.user': 'dendi'}
        count_before = orion_db._get_database().count('experiments')
        count_query = orion_db._get_database().count('experiments', filt)
        # call interface
        assert orion_db.write('experiments', {'pool_size': 16}, filt) == count_query
        database = orion_db._get_database()._db
        assert database['experiments'].count() == count_before
        value = list(database['experiments'].find({}))
        assert value[0]['pool_size'] == 16
        assert value[1]['pool_size'] == 2
        assert value[2]['pool_size'] == 2
        assert value[3]['pool_size'] == 2

    def test_update_with_id(self, exp_config, orion_db):
        """Query using ``_id`` key."""
        filt = {'_id': exp_config[0][1]['_id']}
        count_before = orion_db._get_database().count('experiments')
        # call interface
        assert orion_db.write('experiments', {'pool_size': 36}, filt) == 1
        database = orion_db._get_database()._db
        assert database['experiments'].count() == count_before
        value = list(database['experiments'].find())
        assert value[0]['pool_size'] == 2
        assert value[1]['pool_size'] == 36
        assert value[2]['pool_size'] == 2


@pytest.mark.usefixtures("clean_db")
class TestReadAndWrite(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.read_and_write`."""

    def test_read_and_write_one(self, orion_db, exp_config):
        """Should read and update a single entry in the collection."""
        # Make sure there is only one match
        documents = orion_db.read(
            'experiments',
            {'name': 'supernaedo4'})
        assert len(documents) == 1

        # Find and update atomically
        loaded_config = orion_db.read_and_write(
            'experiments',
            {'name': 'supernaedo4'},
            {'pool_size': 'lalala'})
        exp_config[0][3]['pool_size'] = 'lalala'
        backward.populate_space(loaded_config)
        assert loaded_config == exp_config[0][3]

    def test_read_and_write_many(self, orion_db, exp_config):
        """Should update only one entry."""
        documents = orion_db.read('experiments', {'metadata.user': 'tsirif'})
        assert len(documents) > 1

        # Find many and update first one only
        loaded_config = orion_db.read_and_write(
            'experiments',
            {'metadata.user': 'tsirif'},
            {'pool_size': 'lalala'})

        exp_config[0][1]['pool_size'] = 'lalala'
        backward.populate_space(loaded_config)
        assert loaded_config == exp_config[0][1]

        # Make sure it only changed the first document found
        documents = orion_db.read('experiments', {'metadata.user': 'tsirif'})
        assert documents[0]['pool_size'] == 'lalala'
        assert documents[1]['pool_size'] != 'lalala'

    def test_read_and_write_no_match(self, orion_db):
        """Should return None when there is no match."""
        loaded_config = orion_db.read_and_write(
            'experiments',
            {'name': 'lalala'},
            {'pool_size': 'lalala'})

        assert loaded_config is None

    def test_logging_when_getting_file_lock(self, caplog, orion_db):
        """When logging.level is ERROR, there should be no logging."""
        logging.basicConfig(level=logging.INFO)
        caplog.clear()
        caplog.set_level(logging.ERROR)
        # any operation will trigger the lock.
        orion_db.read(
            'experiments',
            {'name': 'supernaedo2', 'metadata.user': 'dendi'})

        assert 'acquired on orion_db.pkl.lock' not in caplog.text


@pytest.mark.usefixtures("clean_db")
class TestRemove(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.remove`."""

    def test_remove_many_default(self, exp_config, orion_db):
        """Should match existing entries, and delete them all."""
        filt = {'metadata.user': 'tsirif'}
        database = orion_db._get_database()._db
        count_before = database['experiments'].count()
        count_filt = database['experiments'].count(filt)
        # call interface
        assert orion_db.remove('experiments', filt) == count_filt
        database = orion_db._get_database()._db
        assert database['experiments'].count() == count_before - count_filt
        assert database['experiments'].count() == 1
        loaded_config = list(database['experiments'].find())
        backward.populate_space(loaded_config[0])
        assert loaded_config == [exp_config[0][0]]

    def test_remove_with_id(self, exp_config, orion_db):
        """Query using ``_id`` key."""
        filt = {'_id': exp_config[0][0]['_id']}

        database = orion_db._get_database()._db
        count_before = database['experiments'].count()
        # call interface
        assert orion_db.remove('experiments', filt) == 1
        database = orion_db._get_database()._db
        assert database['experiments'].count() == count_before - 1
        loaded_configs = database['experiments'].find()
        for loaded_config in loaded_configs:
            backward.populate_space(loaded_config)
        assert loaded_configs == exp_config[0][1:]

    def test_remove_update_indexes(self, exp_config, orion_db):
        """Verify that indexes are properly update after deletion."""
        with pytest.raises(DuplicateKeyError):
            orion_db.write('experiments', {'_id': exp_config[0][0]['_id']})
        with pytest.raises(DuplicateKeyError):
            orion_db.write('experiments', {'_id': exp_config[0][1]['_id']})

        filt = {'_id': exp_config[0][0]['_id']}

        database = orion_db._get_database()._db
        count_before = database['experiments'].count()
        # call interface
        assert orion_db.remove('experiments', filt) == 1
        database = orion_db._get_database()._db
        assert database['experiments'].count() == count_before - 1
        # Should not fail now, otherwise it means the indexes were not updated properly during
        # remove()
        orion_db.write('experiments', filt)
        # And this should still fail
        with pytest.raises(DuplicateKeyError):
            orion_db.write('experiments', {'_id': exp_config[0][1]['_id']})


@pytest.mark.usefixtures("clean_db")
class TestCount(object):
    """Calls :meth:`orion.core.io.database.pickleddb.PickledDB.count`."""

    def test_count_default(self, exp_config, orion_db):
        """Call just with collection name."""
        found = orion_db.count('trials')
        assert found == len(exp_config[1])

    def test_count_query(self, exp_config, orion_db):
        """Call with a query."""
        found = orion_db.count('trials', {'status': 'completed'})
        assert found == len([x for x in exp_config[1] if x['status'] == 'completed'])

    def test_count_query_with_id(self, exp_config, orion_db):
        """Call querying with unique _id."""
        found = orion_db.count('trials', {'_id': exp_config[1][2]['_id']})
        assert found == 1

    def test_count_nothing(self, orion_db):
        """Call with argument that will not find anything."""
        found = orion_db.count('experiments', {'name': 'lalalanotfound'})
        assert found == 0


def write(field, i):
    """Write the given value to the pickled db."""
    PickledDB.instance = None
    orion_db = PickledDB(host='orion_db.pkl')
    try:
        orion_db.write('concurrent', {field: i})
    except DuplicateKeyError:
        print('dup')
        pass

    print(field, i)


@pytest.mark.usefixtures("clean_db")
class TestConcurreny(object):
    """Test concurrent operations"""

    def test_concurrent_writes(self, orion_db):
        """Test that concurrent writes all get written properly"""
        orion_db.ensure_index('concurrent', 'diff')

        assert orion_db.count('concurrent', {'diff': {'$gt': -1}}) == 0

        Pool(10).starmap(write, (('diff', i) for i in range(10)))

        assert orion_db.count('concurrent', {'diff': {'$gt': -1}}) == 10

    def test_concurrent_unique_writes(self, orion_db):
        """Test that concurrent writes cannot duplicate unique fields"""
        orion_db.ensure_index('concurrent', 'unique', unique=True)

        assert orion_db.count('concurrent', {'unique': 1}) == 0

        Pool(10).starmap(write, (('unique', 1) for i in range(10)))

        assert orion_db.count('concurrent', {'unique': 1}) == 1


def test_empty_file(orion_db):
    """Check that db loading can handle empty files"""
    with open(orion_db.host, 'wb') as f:
        f.write(b'')
    orion_db._get_database()


def test_unpickable_error_find_document():
    """Check error messages for pickledb"""
    class UnpickableClass:
        i_am_not_pickable = None

    unpickable_doc = {
        '_id': 2,
        'a_pickable': 1,
        'b_unpickable': UnpickableClass(),
        'c_pickable': 3
    }

    def make_pickable(uid):
        return {
            '_id': uid,
            'a_pickable': 1,
            'b_pickable': 2,
            'c_pickable': 3
        }

    unpickable_dict_of_dict = [
        make_pickable(1),
        unpickable_doc,
        make_pickable(3)
    ]

    pickable_dict_of_dict = [
        make_pickable(1),
        make_pickable(2),
        make_pickable(3)
    ]

    unpickable_collection = EphemeralCollection()
    unpickable_collection.insert_many(unpickable_dict_of_dict)

    pickable_collection = EphemeralCollection()
    pickable_collection.insert_many(pickable_dict_of_dict)

    database = {
        'pickable_collection': pickable_collection,
        'unpickable_collection': unpickable_collection
    }

    collection, doc = find_unpickable_doc(database)
    assert collection == 'unpickable_collection', 'should return the unpickable document'

    key, value = find_unpickable_field(doc)
    assert key == 'b_unpickable', 'should return the unpickable field'
    assert isinstance(value, UnpickableClass), 'should return the unpickable value'


def test_query_timeout(monkeypatch, orion_db):
    """Verify that filelock.Timeout is catched and reraised as DatabaseTimeout"""
    orion_db.timeout = 0.1

    def never_acquire(self, *arg, **kwargs):
        """Do not try to acquire, raise timeout"""
        raise Timeout(self)

    monkeypatch.setattr(FileLock, 'acquire', never_acquire)

    with pytest.raises(DatabaseTimeout) as exc:
        orion_db.read('whatever', {'it should': 'fail'})

    assert exc.match('Could not acquire lock for PickledDB after 0.1 seconds.')
