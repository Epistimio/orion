#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.io.database.pickleddb`."""
import copy
import logging
import os
import random
import uuid
from datetime import datetime
from multiprocessing import Pool

import pytest
from filelock import FileLock, SoftFileLock, Timeout

import orion.core.utils.backward as backward
from orion.core.io.database import Database, DatabaseError, DatabaseTimeout, DuplicateKeyError
from orion.core.io.database.ephemeraldb import (
    EphemeralCollection,
    EphemeralDB,
    EphemeralDocument,
)
from orion.core.io.database.pickleddb import (
    PickledDB,
    _create_lock,
    find_unpickable_doc,
    find_unpickable_field,
    local_file_systems,
)


ephemeraldb_only = pytest.mark.db_types_only(["ephemeraldb"])


pickleddb_only = pytest.mark.db_types_only(["pickleddb"])


# _DATABASE_PARAMS = ["ephemeraldb", "mongodb", "pickleddb"]
_DB_TYPES = ["ephemeraldb", "pickleddb"]


@pytest.fixture(scope="module", params=_DB_TYPES)
def db_type(request):
    """Return a string identifier of the database types"""
    yield request.param


@pytest.fixture(scope="module")
def orion_db(db_type):
    """Return a EphemeralDB, MongoDB or a PickledDB wrapper instance initiated
       with test opts."""
    if db_type == "ephemeraldb":
        EphemeralDB.instance = None
        orion_db = EphemeralDB()
    elif db_type == "mongodb":
        pass
    elif db_type == "pickleddb":
        PickledDB.instance = None
        orion_db = PickledDB(host="orion_db.pkl")
    else:
        raise ValueError("Invalid database type")

    yield orion_db


@pytest.fixture()
def init_db(orion_db, exp_config):
    """Initialise the database with clean insert example experiment entries to collections."""
    if isinstance(orion_db, EphemeralDB):
        pass
    # elif isinstance(orion_db, MongoDB):
    #     pass
    elif isinstance(orion_db, PickledDB):
        if os.path.exists(orion_db.host):
            os.remove(orion_db.host)

    database = get_db(orion_db)

    database["experiments"].drop()
    database["experiments"].insert_many(exp_config[0])
    database["trials"].drop()
    database["trials"].insert_many(exp_config[1])
    database["workers"].drop()
    database["workers"].insert_many(exp_config[2])
    database["resources"].drop()
    database["resources"].insert_many(exp_config[3])

    dump_db(orion_db, database)

    yield copy.deepcopy(database)


@pytest.fixture()
def clean_db(orion_db, init_db):
    """Clean database for test."""
    print("\n--CLEAN DB {}--".format(type(orion_db)), end="")
    print("\n--CLEAN DB {}--".format(type(init_db)), end="")

    yield

    # Restaure initial database
    if isinstance(orion_db, EphemeralDB):
        orion_db._db = init_db
    # elif isinstance(orion_db, MongoDB):
    #     pass
    elif isinstance(orion_db, PickledDB):
        ephemeral_db = orion_db._get_database()
        ephemeral_db._db = init_db
        orion_db._dump_database(ephemeral_db)


@pytest.fixture(autouse=True)
def skip_if_not_db_type(request, db_type):
    """Skip test if th database type does no match the "db_types_only" marker
    """
    db_types_only = request.node.get_closest_marker("db_types_only")
    if db_types_only and db_type not in db_types_only.args[0]:
        pytest.skip("{} test only".format(db_types_only.args[0]))


# EPHEMERALDB ONLY FIXTURES


@pytest.fixture()
def document(db_type):
    """Return EphemeralDocument."""
    if db_type != "ephemeraldb":
        pytest.skip("ephemeraldb test only")
    yield EphemeralDocument({"_id": 1, "hello": "there", "mighty": "duck"})


@pytest.fixture()
def subdocument(db_type):
    """Return EphemeralDocument with a subdocument."""
    if db_type != "ephemeraldb":
        pytest.skip("ephemeraldb test only")
    yield EphemeralDocument(
        {"_id": 1, "hello": "there", "mighty": "duck", "and": {"the": "drake"}}
    )


@pytest.fixture()
def collection(document, db_type):
    """Return EphemeralCollection."""
    if db_type != "ephemeraldb":
        pytest.skip("ephemeraldb test only")

    collection = EphemeralCollection()
    collection.insert_many([document.to_dict()])

    yield collection


def get_db(orion_db):
    if isinstance(orion_db, EphemeralDB):
        return orion_db._db
    # elif isinstance(orion_db, MongoDB):
    #     return orion_db
    elif isinstance(orion_db, PickledDB):
        return orion_db._get_database()._db
    else:
        raise TypeError("Invalid database type")


def dump_db(orion_db, db):
    if isinstance(orion_db, EphemeralDB):
        pass
    # elif isinstance(orion_db, MongoDB):
    #     pass
    elif isinstance(orion_db, PickledDB):
        ephemeral_db = orion_db._get_database()
        ephemeral_db._db = db
        orion_db._dump_database(ephemeral_db)
    else:
        raise TypeError("Invalid database type")


@pytest.mark.usefixtures("clean_db")
class TestEnsureIndex(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.ensure_index`."""

    def test_new_index(self, orion_db):
        """Index should be added to pickled database"""
        assert (
                "new_field_1" not in get_db(orion_db)["new_collection"]._indexes
        )

        orion_db.ensure_index("new_collection", "new_field", unique=False)
        assert (
                "new_field_1" not in get_db(orion_db)["new_collection"]._indexes
        )

        orion_db.ensure_index("new_collection", "new_field", unique=True)
        assert "new_field_1" in get_db(orion_db)["new_collection"]._indexes

    def test_existing_index(self, orion_db):
        """Index should be added to pickled database and reattempt should do nothing"""
        assert (
            "new_field_1" not in get_db(orion_db)["new_collection"]._indexes
        )

        orion_db.ensure_index("new_collection", "new_field", unique=True)
        assert "new_field_1" in get_db(orion_db)["new_collection"]._indexes

        # reattempt
        orion_db.ensure_index("new_collection", "new_field", unique=True)
        assert "new_field_1" in get_db(orion_db)["new_collection"]._indexes

    def test_compound_index(self, orion_db):
        """Tuple of Index should be added as a compound index."""
        assert (
            "name_1_metadata.user_1"
            not in get_db(orion_db)["experiments"]._indexes
        )
        orion_db.ensure_index(
            "experiments",
            [("name", Database.ASCENDING), ("metadata.user", Database.ASCENDING)],
            unique=True,
        )
        assert (
            "name_1_metadata.user_1"
            in get_db(orion_db)["experiments"]._indexes
        )


@pytest.mark.usefixtures("clean_db")
class TestRead(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.read`."""

    def test_read_experiment(self, exp_config, orion_db):
        """Fetch a whole experiment entries."""
        loaded_config = orion_db.read(
            "trials", {"experiment": "supernaedo2-dendi", "status": "new"}
        )
        assert loaded_config == [exp_config[1][3], exp_config[1][4]]

        loaded_config = orion_db.read(
            "trials",
            {
                "experiment": "supernaedo2-dendi",
                "submit_time": exp_config[1][3]["submit_time"],
            },
        )
        assert loaded_config == [exp_config[1][3]]
        assert loaded_config[0]["_id"] == exp_config[1][3]["_id"]

    def test_read_with_id(self, exp_config, orion_db):
        """Query using ``_id`` key."""
        loaded_config = orion_db.read("experiments", {"_id": exp_config[0][2]["_id"]})
        backward.populate_space(loaded_config[0])
        assert loaded_config == [exp_config[0][2]]

    def test_read_default(self, exp_config, orion_db):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            "experiments",
            {"name": "supernaedo2", "metadata.user": "tsirif"},
            selection={"algorithms": 1, "_id": 0},
        )
        assert value == [{"algorithms": exp_config[0][0]["algorithms"]}]

    def test_read_nothing(self, orion_db):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            "experiments",
            {"name": "not_found", "metadata.user": "tsirif"},
            selection={"algorithms": 1},
        )
        assert value == []

    def test_read_trials(self, exp_config, orion_db):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            "trials",
            {
                "experiment": "supernaedo2-dendi",
                "submit_time": {"$gte": datetime(2017, 11, 23, 0, 0, 0)},
            },
        )
        assert value == [exp_config[1][1]] + exp_config[1][3:7]

        value = orion_db.read(
            "trials",
            {
                "experiment": "supernaedo2-dendi",
                "submit_time": {"$gt": datetime(2017, 11, 23, 0, 0, 0)},
            },
        )
        assert value == exp_config[1][3:7]

    def test_null_comp(self, exp_config, orion_db):
        """Fetch value(s) from an entry."""
        all_values = orion_db.read(
            "trials",
            {
                "experiment": "supernaedo2-dendi",
                "end_time": {"$gte": datetime(2017, 11, 1, 0, 0, 0)},
            },
        )

        db = get_db(orion_db)
        db["trials"]._documents[0]._data["end_time"] = None
        dump_db(orion_db, db)

        values = orion_db.read(
            "trials",
            {
                "experiment": "supernaedo2-dendi",
                "end_time": {"$gte": datetime(2017, 11, 1, 0, 0, 0)},
            },
        )
        assert len(values) == len(all_values) - 1


@pytest.mark.usefixtures("clean_db")
class TestWrite(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.write`."""

    def test_insert_one(self, orion_db):
        """Should insert a single new entry in the collection."""
        item = {"exp_name": "supernaekei", "user": "tsirif"}
        count_before = orion_db.count("experiments")
        # call interface
        assert orion_db.write("experiments", item) == 1
        assert orion_db.count("experiments") == count_before + 1
        value = (
            get_db(orion_db)["experiments"]
            .find({"exp_name": "supernaekei"})[0]
        )
        assert value == item

    def test_insert_many(self, orion_db):
        """Should insert two new entry (as a list) in the collection."""
        item = [
            {"exp_name": "supernaekei2", "user": "tsirif"},
            {"exp_name": "supernaekei3", "user": "tsirif"},
        ]
        count_before = get_db(orion_db)["experiments"].count()
        # call interface
        assert orion_db.write("experiments", item) == 2
        database = get_db(orion_db)
        assert database["experiments"].count() == count_before + 2
        value = database["experiments"].find({"exp_name": "supernaekei2"})[0]
        assert value == item[0]
        value = database["experiments"].find({"exp_name": "supernaekei3"})[0]
        assert value == item[1]

    def test_update_many_default(self, orion_db):
        """Should match existing entries, and update some of their keys."""
        filt = {"metadata.user": "dendi"}
        count_before = orion_db.count("experiments")
        count_query = orion_db.count("experiments", filt)
        # call interface
        assert orion_db.write("experiments", {"pool_size": 16}, filt) == count_query
        database = get_db(orion_db)
        assert database["experiments"].count() == count_before
        value = list(database["experiments"].find({}))
        assert value[0]["pool_size"] == 16
        assert value[1]["pool_size"] == 2
        assert value[2]["pool_size"] == 2
        assert value[3]["pool_size"] == 2

    def test_update_with_id(self, exp_config, orion_db):
        """Query using ``_id`` key."""
        filt = {"_id": exp_config[0][1]["_id"]}
        count_before = orion_db.count("experiments")
        # call interface
        assert orion_db.write("experiments", {"pool_size": 36}, filt) == 1
        database = get_db(orion_db)
        assert database["experiments"].count() == count_before
        value = list(database["experiments"].find())
        assert value[0]["pool_size"] == 2
        assert value[1]["pool_size"] == 36
        assert value[2]["pool_size"] == 2


@pytest.mark.usefixtures("clean_db")
class TestReadAndWrite(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.read_and_write`."""

    def test_read_and_write_one(self, orion_db, exp_config):
        """Should read and update a single entry in the collection."""
        # Make sure there is only one match
        documents = orion_db.read("experiments", {"name": "supernaedo4"})
        assert len(documents) == 1

        # Find and update atomically
        loaded_config = orion_db.read_and_write(
            "experiments", {"name": "supernaedo4"}, {"pool_size": "lalala"}
        )
        exp_config[0][3]["pool_size"] = "lalala"
        backward.populate_space(loaded_config)
        assert loaded_config == exp_config[0][3]

    def test_read_and_write_many(self, orion_db, exp_config):
        """Should update only one entry."""
        documents = orion_db.read("experiments", {"metadata.user": "tsirif"})
        assert len(documents) > 1

        # Find many and update first one only
        loaded_config = orion_db.read_and_write(
            "experiments", {"metadata.user": "tsirif"}, {"pool_size": "lalala"}
        )

        exp_config[0][1]["pool_size"] = "lalala"
        backward.populate_space(loaded_config)
        assert loaded_config == exp_config[0][1]

        # Make sure it only changed the first document found
        documents = orion_db.read("experiments", {"metadata.user": "tsirif"})
        assert documents[0]["pool_size"] == "lalala"
        assert documents[1]["pool_size"] != "lalala"

    def test_read_and_write_no_match(self, orion_db):
        """Should return None when there is no match."""
        loaded_config = orion_db.read_and_write(
            "experiments", {"name": "lalala"}, {"pool_size": "lalala"}
        )

        assert loaded_config is None

    def test_logging_when_getting_file_lock(self, caplog, orion_db):
        """When logging.level is ERROR, there should be no logging."""
        logging.basicConfig(level=logging.INFO)
        caplog.clear()
        caplog.set_level(logging.ERROR)
        # any operation will trigger the lock.
        orion_db.read("experiments", {"name": "supernaedo2", "metadata.user": "dendi"})

        assert "acquired on orion_db.pkl.lock" not in caplog.text


@pytest.mark.usefixtures("clean_db")
class TestRemove(object):
    """Calls to :meth:`orion.core.io.database.pickleddb.PickledDB.remove`."""

    def test_remove_many_default(self, exp_config, orion_db):
        """Should match existing entries, and delete them all."""
        filt = {"metadata.user": "tsirif"}
        database = get_db(orion_db)
        count_before = database["experiments"].count()
        count_filt = database["experiments"].count(filt)
        # call interface
        assert orion_db.remove("experiments", filt) == count_filt
        database = get_db(orion_db)
        assert database["experiments"].count() == count_before - count_filt
        assert database["experiments"].count() == 1
        loaded_config = list(database["experiments"].find())
        backward.populate_space(loaded_config[0])
        assert loaded_config == [exp_config[0][0]]

    def test_remove_with_id(self, exp_config, orion_db):
        """Query using ``_id`` key."""
        filt = {"_id": exp_config[0][0]["_id"]}

        database = get_db(orion_db)
        count_before = database["experiments"].count()
        # call interface
        assert orion_db.remove("experiments", filt) == 1
        database = get_db(orion_db)
        assert database["experiments"].count() == count_before - 1
        loaded_configs = database["experiments"].find()
        for loaded_config in loaded_configs:
            backward.populate_space(loaded_config)
        assert loaded_configs == exp_config[0][1:]

    def test_remove_update_indexes(self, exp_config, orion_db):
        """Verify that indexes are properly update after deletion."""
        with pytest.raises(DuplicateKeyError):
            orion_db.write("experiments", {"_id": exp_config[0][0]["_id"]})
        with pytest.raises(DuplicateKeyError):
            orion_db.write("experiments", {"_id": exp_config[0][1]["_id"]})

        filt = {"_id": exp_config[0][0]["_id"]}

        database = get_db(orion_db)
        count_before = database["experiments"].count()
        # call interface
        assert orion_db.remove("experiments", filt) == 1
        database = get_db(orion_db)
        assert database["experiments"].count() == count_before - 1
        # Should not fail now, otherwise it means the indexes were not updated properly during
        # remove()
        orion_db.write("experiments", filt)
        # And this should still fail
        with pytest.raises(DuplicateKeyError):
            orion_db.write("experiments", {"_id": exp_config[0][1]["_id"]})


@pytest.mark.usefixtures("clean_db")
class TestCount(object):
    """Calls :meth:`orion.core.io.database.pickleddb.PickledDB.count`."""

    def test_count_default(self, exp_config, orion_db):
        """Call just with collection name."""
        found = orion_db.count("trials")
        assert found == len(exp_config[1])

    def test_count_query(self, exp_config, orion_db):
        """Call with a query."""
        found = orion_db.count("trials", {"status": "completed"})
        assert found == len([x for x in exp_config[1] if x["status"] == "completed"])

    def test_count_query_with_id(self, exp_config, orion_db):
        """Call querying with unique _id."""
        found = orion_db.count("trials", {"_id": exp_config[1][2]["_id"]})
        assert found == 1

    def test_count_nothing(self, orion_db):
        """Call with argument that will not find anything."""
        found = orion_db.count("experiments", {"name": "lalalanotfound"})
        assert found == 0


@pytest.mark.usefixtures("clean_db")
class TestIndexInformation(object):
    """Calls :meth:`orion.core.io.database.mongodb.EphemeralDB.count`."""

    def test_no_index(self, orion_db):
        """Test that no index is returned when there is none."""
        assert orion_db.index_information("experiments") == {"_id_": True}

    def test_single_index(self, orion_db):
        """Test that single indexes are ignored if not unique."""
        orion_db.ensure_index("experiments", [("name", EphemeralDB.ASCENDING)])

        assert orion_db.index_information("experiments") == {"_id_": True}

    def test_single_index_unique(self, orion_db):
        """Test with single unique indexes."""
        orion_db.ensure_index(
            "experiments", [("name", EphemeralDB.ASCENDING)], unique=True
        )

        assert orion_db.index_information("experiments") == {
            "_id_": True,
            "name_1": True,
        }

    def test_ordered_index(self, orion_db):
        """Test that ordered indexes are not taken into account."""
        orion_db.ensure_index(
            "experiments", [("name", EphemeralDB.DESCENDING)], unique=True
        )

        assert orion_db.index_information("experiments") == {
            "_id_": True,
            "name_1": True,
        }

    def test_compound_index(self, orion_db):
        """Test representation of compound indexes."""
        orion_db.ensure_index(
            "experiments",
            [("name", EphemeralDB.DESCENDING), ("version", EphemeralDB.ASCENDING)],
            unique=True,
        )

        index_info = orion_db.index_information("experiments")
        assert index_info == {"_id_": True, "name_1_version_1": True}


@pytest.mark.usefixtures("clean_db")
class TestDropIndex(object):
    """Calls :meth:`orion.core.io.database.mongodb.EphemeralDB.count`."""

    def test_no_index(self, orion_db):
        """Test that no index is returned when there is none."""
        with pytest.raises(DatabaseError) as exc:
            orion_db.drop_index("experiments", "i_dont_exist")
        assert "index not found with name" in str(exc.value)

    def test_drop_single_index(self, orion_db):
        """Test with single indexes."""
        orion_db.ensure_index(
            "experiments", [("name", EphemeralDB.ASCENDING)], unique=True
        )
        assert orion_db.index_information("experiments") == {
            "_id_": True,
            "name_1": True,
        }
        orion_db.drop_index("experiments", "name_1")
        assert orion_db.index_information("experiments") == {"_id_": True}

    def test_drop_ordered_single_index(self, orion_db):
        """Test with single indexes."""
        orion_db.ensure_index(
            "experiments", [("name", EphemeralDB.DESCENDING)], unique=True
        )
        index_info = orion_db.index_information("experiments")
        assert index_info == {"_id_": True, "name_1": True}
        orion_db.drop_index("experiments", "name_1")
        index_info = orion_db.index_information("experiments")
        assert index_info == {"_id_": True}

    def test_drop_ordered_compound_index(self, orion_db):
        """Test with single indexes."""
        orion_db.ensure_index(
            "experiments",
            [("name", EphemeralDB.ASCENDING), ("version", EphemeralDB.DESCENDING)],
            unique=True,
        )
        index_info = orion_db.index_information("experiments")
        assert index_info == {"_id_": True, "name_1_version_1": True}
        orion_db.drop_index("experiments", "name_1_version_1")
        index_info = orion_db.index_information("experiments")
        assert index_info == {"_id_": True}


# EPHEMERALDB ONLY TESTS


@ephemeraldb_only
@pytest.mark.usefixtures("clean_db")
class TestIndex(object):
    """Test index for :meth:`orion.core.io.database.ephemeraldb.EphemeralCollection`."""

    def test_create_index(self, collection):
        """Test if new index added property."""
        collection.create_index("hello")
        assert collection._indexes == {"_id_": (("_id",), {(1,)})}

        collection.create_index("hello", unique=True)
        assert collection._indexes == {
            "_id_": (("_id",), {(1,)}),
            "hello_1": (("hello",), {("there",)}),
        }

    def test_track_index(self, collection):
        """Test if index values are tracked property."""
        collection.create_index("hello", unique=True)
        collection.insert_many([{"hello": "here"}, {"hello": 2}])
        assert collection._indexes == {
            "_id_": (("_id",), {(1,), (2,), (3,)}),
            "hello_1": (("hello",), {("there",), ("here",), (2,)}),
        }

    def test_index_over_non_existing_field(self, collection):
        """Test if index values are tracked property."""
        collection.create_index(
            [("hello", EphemeralDB.DESCENDING), ("idontexist", EphemeralDB.ASCENDING)],
            unique=True,
        )

        collection.insert_many([{"hello": "here"}, {"hello": 2}])
        assert collection._indexes == {
            "_id_": (("_id",), {(1,), (2,), (3,)}),
            "hello_1_idontexist_1": (
                ("hello", "idontexist"),
                {("there", None), ("here", None), (2, None)},
            ),
        }
        assert collection.find({}, selection={"hello": 1, "idontexist": 1}) == [
            {"_id": 1, "hello": "there", "idontexist": None},
            {"_id": 2, "hello": "here", "idontexist": None},
            {"_id": 3, "hello": 2, "idontexist": None},
        ]


@ephemeraldb_only
@pytest.mark.usefixtures("clean_db")
class TestSelect(object):
    """Calls :meth:`orion.core.io.database.ephemeraldb.EphemeralDocument.select`."""

    def test_select_all(self, document):
        """Select only one field."""
        assert document.select({}) == {"_id": 1, "hello": "there", "mighty": "duck"}

    def test_select_id(self, document):
        """Select only one field."""
        assert document.select({"_id": 1}) == {"_id": 1}

    def test_select_one(self, document):
        """Select only one field."""
        assert document.select({"hello": 1}) == {"_id": 1, "hello": "there"}

    def test_select_two(self, document):
        """Select only two field."""
        assert document.select({"hello": 1, "mighty": 1}) == {
            "_id": 1,
            "hello": "there",
            "mighty": "duck",
        }

    def test_unselect_one(self, document):
        """Unselect only one field."""
        assert document.select({"hello": 0}) == {"_id": 1, "mighty": "duck"}

    def test_unselect_two(self, document):
        """Unselect two field."""
        assert document.select({"_id": 0, "hello": 0}) == {"mighty": "duck"}

    def test_mixed_select(self, document):
        """Select one field and unselect _id."""
        assert document.select({"_id": 0, "hello": 1}) == {"hello": "there"}

    def test_select_unexisting_field(self, document):
        """Select field that does not exist and should return None."""
        assert document.select({"idontexist": 1}) == {"_id": 1, "idontexist": None}


@ephemeraldb_only
@pytest.mark.usefixtures("clean_db")
class TestMatch:
    """Calls :meth:`orion.core.io.database.ephemeraldb.EphemeralDocument.match`."""

    def test_match_eq(self, document):
        """Test eq operator"""
        assert document.match({"hello": "there"})
        assert not document.match({"hello": "not there"})

    def test_match_sub_eq(self, subdocument):
        """Test eq operator with sub document"""
        assert subdocument.match({"and.the": "drake"})
        assert not subdocument.match({"and.no": "drake"})

    def test_match_in(self, subdocument):
        """Test $in operator with document"""
        assert subdocument.match({"hello": {"$in": ["there", "here"]}})
        assert not subdocument.match({"hello": {"$in": ["ici", "here"]}})

    def test_match_sub_in(self, subdocument):
        """Test $in operator with sub document"""
        assert subdocument.match({"and.the": {"$in": ["duck", "drake"]}})
        assert not subdocument.match({"and.the": {"$in": ["hyppo", "lion"]}})

    def test_match_gte(self, document):
        """Test $gte operator with document"""
        assert document.match({"_id": {"$gte": 1}})
        assert document.match({"_id": {"$gte": 0}})
        assert not document.match({"_id": {"$gte": 2}})

    def test_match_gt(self, document):
        """Test $gt operator with document"""
        assert document.match({"_id": {"$gt": 0}})
        assert not document.match({"_id": {"$gt": 1}})

    def test_match_lte(self, document):
        """Test $lte operator with document"""
        assert document.match({"_id": {"$lte": 2}})
        assert document.match({"_id": {"$lte": 1}})
        assert not document.match({"_id": {"$lte": 0}})

    def test_match_ne(self, document):
        """Test $ne operator with document"""
        assert document.match({"hello": {"$ne": "here"}})
        assert not document.match({"hello": {"$ne": "there"}})

    def test_match_bad_operator(self, document):
        """Test invalid operator handling"""
        with pytest.raises(ValueError) as exc:
            document.match({"_id": {"$voici_voila": 0}})

        assert "Operator '$voici_voila' is not supported" in str(exc.value)


# PICKLEDDB ONLY TESTS


def write(field, i):
    """Write the given value to the pickled db."""
    PickledDB.instance = None
    orion_db = PickledDB(host="orion_db.pkl")
    try:
        orion_db.write("concurrent", {field: i})
    except DuplicateKeyError:
        print("dup")
        pass

    print(field, i)


@pickleddb_only
@pytest.mark.usefixtures("clean_db")
class TestConcurreny(object):
    """Test concurrent operations"""

    def test_concurrent_writes(self, orion_db):
        """Test that concurrent writes all get written properly"""
        orion_db.ensure_index("concurrent", "diff")

        assert orion_db.count("concurrent", {"diff": {"$gt": -1}}) == 0

        Pool(10).starmap(write, (("diff", i) for i in range(10)))

        assert orion_db.count("concurrent", {"diff": {"$gt": -1}}) == 10

    def test_concurrent_unique_writes(self, orion_db):
        """Test that concurrent writes cannot duplicate unique fields"""
        orion_db.ensure_index("concurrent", "unique", unique=True)

        assert orion_db.count("concurrent", {"unique": 1}) == 0

        Pool(10).starmap(write, (("unique", 1) for i in range(10)))

        assert orion_db.count("concurrent", {"unique": 1}) == 1


@pickleddb_only
def test_empty_file(orion_db):
    """Check that db loading can handle empty files"""
    with open(orion_db.host, "wb") as f:
        f.write(b"")
    get_db(orion_db)


@pickleddb_only
def test_unpickable_error_find_document():
    """Check error messages for pickledb"""

    class UnpickableClass:
        i_am_not_pickable = None

    unpickable_doc = {
        "_id": 2,
        "a_pickable": 1,
        "b_unpickable": UnpickableClass(),
        "c_pickable": 3,
    }

    def make_pickable(uid):
        return {"_id": uid, "a_pickable": 1, "b_pickable": 2, "c_pickable": 3}

    unpickable_dict_of_dict = [make_pickable(1), unpickable_doc, make_pickable(3)]

    pickable_dict_of_dict = [make_pickable(1), make_pickable(2), make_pickable(3)]

    unpickable_collection = EphemeralCollection()
    unpickable_collection.insert_many(unpickable_dict_of_dict)

    pickable_collection = EphemeralCollection()
    pickable_collection.insert_many(pickable_dict_of_dict)

    database = {
        "pickable_collection": pickable_collection,
        "unpickable_collection": unpickable_collection,
    }

    collection, doc = find_unpickable_doc(database)
    assert (
        collection == "unpickable_collection"
    ), "should return the unpickable document"

    key, value = find_unpickable_field(doc)
    assert key == "b_unpickable", "should return the unpickable field"
    assert isinstance(value, UnpickableClass), "should return the unpickable value"


@pickleddb_only
def test_query_timeout(monkeypatch, orion_db):
    """Verify that filelock.Timeout is catched and reraised as DatabaseTimeout"""
    orion_db.timeout = 0.1

    def never_acquire(self, *arg, **kwargs):
        """Do not try to acquire, raise timeout"""
        raise Timeout(self)

    monkeypatch.setattr(FileLock, "acquire", never_acquire)

    with pytest.raises(DatabaseTimeout) as exc:
        orion_db.read("whatever", {"it should": "fail"})

    assert exc.match("Could not acquire lock for PickledDB after 0.1 seconds.")


class _MockFS:
    pass


@pickleddb_only
@pytest.mark.parametrize(
    "fs_type,options,file_lock_class",
    [
        ["lustre", [], SoftFileLock],
        ["lustre", ["flock"], FileLock],
        ["lustre", ["localfilelock"], SoftFileLock],
        ["lustre", ["flock", "localflock"], SoftFileLock],
        ["beegfs", [], SoftFileLock],
        ["beegfs", ["tuneUseGlobalFileLocks"], FileLock],
        ["gpfs", [], FileLock],
        ["nfs", [], SoftFileLock],
        ["idontknow", [], SoftFileLock],
    ]
    + [[fs_type, [], FileLock] for fs_type in local_file_systems],
)
def test_file_locks(monkeypatch, fs_type, options, file_lock_class):
    """Verify that the correct file lock type is used based on FS type and configuration"""

    def _get_fs(path):
        fs = _MockFS()

        choices = [str(uuid.uuid4()) for _ in range(random.randint(3, 10))] + options
        random.shuffle(choices)
        fs.opts = ",".join(choices)
        fs.fstype = fs_type
        return fs

    monkeypatch.setattr("orion.core.io.database.pickleddb._get_fs", _get_fs)

    assert isinstance(_create_lock("/whatever/the/path/is"), file_lock_class)
