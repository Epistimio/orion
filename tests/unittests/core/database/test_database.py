#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.io.database.pickleddb`."""
import pickle
from datetime import datetime

import pytest

from orion.core.io.database import Database, DatabaseError, DuplicateKeyError
from orion.core.io.database.ephemeraldb import EphemeralDB
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB

from .conftest import insert_test_collection

_DB_TYPES = ["ephemeraldb", "mongodb", "pickleddb"]


def get_db(orion_db):
    """Get database instance"""
    if isinstance(orion_db, EphemeralDB):
        return orion_db._db
    elif isinstance(orion_db, MongoDB):
        return orion_db._db
    elif isinstance(orion_db, PickledDB):
        return orion_db._get_database()._db
    else:
        raise TypeError("Invalid database type")


def dump_db(orion_db, db):
    """Dump database if necessary"""
    if isinstance(orion_db, EphemeralDB):
        pass
    elif isinstance(orion_db, MongoDB):
        pass
    elif isinstance(orion_db, PickledDB):
        ephemeral_db = orion_db._get_database()
        ephemeral_db._db = db
        orion_db._dump_database(ephemeral_db)
    else:
        raise TypeError("Invalid database type")


# TESTS SET


@pytest.mark.usefixtures("clean_db")
@pytest.mark.drop_collections(["new_collection"])
class TestEnsureIndex:
    """Calls to :meth:`orion.core.io.database.AbstractDB.ensure_index`."""

    @pytest.mark.parametrize(
        "db_test_data",
        [
            {
                "ephemeraldb_pickleddb": (False, "new_field", "new_field_1", False),
                "mongodb": (False, "new_field", "new_field_1", True),
            },
            {
                "ephemeraldb_pickleddb": (True, "new_field", "new_field_1", True),
                "mongodb": (True, "new_field", "new_field_1", True),
            },
        ],
        indirect=True,
    )
    def test_new_index(self, orion_db, db_test_data):
        """Index should be added to database"""
        unique, key, stored_key, key_present = db_test_data
        assert stored_key not in get_db(orion_db)["new_collection"].index_information()

        orion_db.ensure_index("new_collection", key, unique=unique)
        assert (
            stored_key in get_db(orion_db)["new_collection"].index_information()
        ) == key_present

    def test_existing_index(self, orion_db):
        """Index should be added to database and reattempt should do nothing"""
        assert (
            "new_field_1" not in get_db(orion_db)["new_collection"].index_information()
        )

        orion_db.ensure_index("new_collection", "new_field", unique=True)
        assert "new_field_1" in get_db(orion_db)["new_collection"].index_information()

        # reattempt
        orion_db.ensure_index("new_collection", "new_field", unique=True)
        assert "new_field_1" in get_db(orion_db)["new_collection"].index_information()

    @pytest.mark.parametrize(
        "db_test_data",
        [
            {
                "ephemeraldb_pickleddb": ("end_time", "end_time_1"),
                "mongodb": ("end_time", "end_time_-1"),
            }
        ],
        indirect=True,
    )
    def test_ordered_index(self, orion_db, db_test_data):
        """Sort order should only be added to index when executed on a mongo
        database"""
        key, stored_key = db_test_data
        assert stored_key not in get_db(orion_db)["new_collection"].index_information()
        orion_db.ensure_index(
            "new_collection", [(key, Database.DESCENDING)], unique=True
        )
        assert stored_key in get_db(orion_db)["new_collection"].index_information()

    def test_compound_index(self, orion_db):
        """Tuple of Index should be added as a compound index."""
        assert (
            "name_1_metadata.user_1"
            not in get_db(orion_db)["experiments"].index_information()
        )
        orion_db.ensure_index(
            "experiments",
            [("name", Database.ASCENDING), ("metadata.user", Database.ASCENDING)],
            unique=True,
        )
        assert (
            "name_1_metadata.user_1"
            in get_db(orion_db)["experiments"].index_information()
        )


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestRead:
    """Calls to :meth:`orion.core.io.database.AbstractDB.read`."""

    def test_read_entries(self, orion_db, test_collection):
        """Fetch a whole entries."""
        loaded_config = orion_db.read(
            "test_collection", {"field1": "same1", "same_field": "same"}
        )
        assert loaded_config == [test_collection[0], test_collection[2]]

        loaded_config = orion_db.read(
            "test_collection", {"field1": "same1", "unique_field": "unique2"}
        )
        assert loaded_config == test_collection[2:]
        assert loaded_config[0]["_id"] == test_collection[2]["_id"]

    def test_read_with_id(self, orion_db, test_collection):
        """Query using ``_id`` key."""
        loaded_config = orion_db.read("test_collection", {"_id": 1})
        assert loaded_config == test_collection[1:2]

    def test_read_default(self, orion_db, test_collection):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            "test_collection",
            {"field1": "same1", "same_comp.ound": "same_compound"},
            selection={"unique_comp": 1, "_id": 0},
        )
        assert value == [
            {"unique_comp": test_collection[0]["unique_comp"]},
            {"unique_comp": test_collection[2]["unique_comp"]},
        ]

    def test_read_nothing(self, orion_db):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            "experiments",
            {"name": "not_found", "metadata.user": "tsirif"},
            selection={"algorithm": 1},
        )
        assert value == []

    def test_read_trials(self, orion_db, test_collection):
        """Fetch value(s) from an entry."""
        value = orion_db.read(
            "test_collection",
            {
                "same_field": "same",
                "datetime": {"$gte": datetime(2017, 11, 23, 0, 0, 0)},
            },
        )
        assert value == test_collection[1:]

        value = orion_db.read(
            "test_collection",
            {
                "same_field": "same",
                "datetime": {"$gt": datetime(2017, 11, 23, 0, 0, 0)},
            },
        )
        assert value == test_collection[2:]

    @pytest.mark.db_types_only(["ephemeraldb", "pickleddb"])
    def test_null_comp(self, orion_db):
        """Fetch value(s) from an entry."""
        all_values = orion_db.read(
            "test_collection",
            {
                "same_field": "same",
                "datetime": {"$gte": datetime(2017, 11, 1, 0, 0, 0)},
            },
        )

        db = get_db(orion_db)
        db["test_collection"]._documents[0]._data["datetime"] = None
        dump_db(orion_db, db)

        values = orion_db.read(
            "test_collection",
            {
                "same_field": "same",
                "datetime": {"$gte": datetime(2017, 11, 1, 0, 0, 0)},
            },
        )
        assert len(values) == len(all_values) - 1


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestWrite:
    """Calls to :meth:`orion.core.io.database.AbstractDB.write`."""

    def test_insert_one(self, orion_db):
        """Should insert a single new entry in the collection."""
        item = {"exp_name": "supernaekei", "user": "tsirif"}
        count_before = orion_db.count("experiments")
        # call interface
        assert orion_db.write("experiments", item) == 1
        assert orion_db.count("experiments") == count_before + 1
        value = get_db(orion_db)["experiments"].find({"exp_name": "supernaekei"})[0]
        assert value == item

    def test_insert_many(self, orion_db):
        """Should insert two new entry (as a list) in the collection."""
        item = [
            {"exp_name": "supernaekei2", "user": "tsirif"},
            {"exp_name": "supernaekei3", "user": "tsirif"},
        ]
        count_before = orion_db.count("experiments")
        # call interface
        assert orion_db.write("experiments", item) == 2
        database = get_db(orion_db)
        assert orion_db.count("experiments") == count_before + 2
        value = database["experiments"].find({"exp_name": "supernaekei2"})[0]
        assert value == item[0]
        value = database["experiments"].find({"exp_name": "supernaekei3"})[0]
        assert value == item[1]

    def test_update_many_default(self, orion_db):
        """Should match existing entries, and update some of their keys."""
        filt = {"field1": "same1"}
        count_before = orion_db.count("test_collection")
        count_query = orion_db.count("test_collection", filt)
        # call interface
        assert (
            orion_db.write("test_collection", {"same_field": "diff"}, filt)
            == count_query
        )
        database = get_db(orion_db)
        assert orion_db.count("test_collection") == count_before
        value = list(database["test_collection"].find({}))
        assert value[0]["same_field"] == "diff"
        assert value[1]["same_field"] == "same"
        assert value[2]["same_field"] == "diff"

    def test_update_with_id(self, orion_db, test_collection):
        """Query using ``_id`` key."""
        filt = {"_id": test_collection[1]["_id"]}
        count_before = orion_db.count("test_collection")
        # call interface
        assert orion_db.write("test_collection", {"same_field": "diff"}, filt) == 1
        database = get_db(orion_db)
        assert orion_db.count("test_collection") == count_before
        value = list(database["test_collection"].find())
        assert value[0]["same_field"] == "same"
        assert value[1]["same_field"] == "diff"
        assert value[2]["same_field"] == "same"

    def test_insert_with_id_then_without_id(self, orion_db):
        """Insert ``_id`` and test inserts without ids."""

        # First check that custom id can be specified.
        item = {"exp_name": "supernaekei", "user": "tsirif", "_id": "my-custom-id"}
        count_before = orion_db.count("experiments")
        # call interface
        assert orion_db.write("experiments", item) == 1
        assert orion_db.count("experiments") == count_before + 1
        value = get_db(orion_db)["experiments"].find({"exp_name": "supernaekei"})[0]
        assert value == item

        # Now check that custom id will work properly with DB inferred next id.
        # (item2 does not have a _id specified, thus the DB must infer the id to use)
        item2 = dict(item)
        del item2["_id"]
        assert orion_db.write("experiments", item2) == 1
        assert orion_db.count("experiments") == count_before + 2
        values = get_db(orion_db)["experiments"].find({"exp_name": "supernaekei"})
        assert values[0] == item
        item2["_id"] = values[1]["_id"]
        assert values[1] == item2

    def test_no_upsert(self, orion_db):
        """Query with a non-existent ``_id`` should no upsert something."""
        assert (
            orion_db.write("experiments", {"pool_size": 66}, {"_id": "lalalathisisnew"})
            == 0
        )


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestReadAndWrite:
    """Calls to :meth:`orion.core.io.database.AbstractDB.read_and_write`."""

    def test_read_and_write_one(self, orion_db, test_collection):
        """Should read and update a single entry in the collection."""
        # Make sure there is only one match
        documents = orion_db.read("test_collection", {"unique_field": "unique1"})
        assert len(documents) == 1

        # Find and update atomically
        loaded_config = orion_db.read_and_write(
            "test_collection", {"unique_field": "unique1"}, {"field0": "lalala"}
        )
        test_collection[1]["field0"] = "lalala"
        assert loaded_config == test_collection[1]

    def test_read_and_write_many(self, orion_db, test_collection):
        """Should update only one entry."""
        documents = orion_db.read("test_collection", {"same_field": "same"})
        assert len(documents) > 1

        # Find many and update first one only
        loaded_config = orion_db.read_and_write(
            "test_collection", {"same_field": "same"}, {"unique_field": "lalala"}
        )

        test_collection[0]["unique_field"] = "lalala"
        assert loaded_config == test_collection[0]

        # Make sure it only changed the first document found
        documents = orion_db.read("test_collection", {"same_field": "same"})
        assert documents[0]["unique_field"] == "lalala"
        assert documents[1]["unique_field"] != "lalala"

    def test_read_and_write_no_match(self, orion_db):
        """Should return None when there is no match."""
        loaded_config = orion_db.read_and_write(
            "experiments", {"name": "lalala"}, {"pool_size": "lalala"}
        )

        assert loaded_config is None


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestRemove:
    """Calls to :meth:`orion.core.io.database.AbstractDB.remove`."""

    def test_remove_many_default(self, orion_db, test_collection):
        """Should match existing entries, and delete them all."""
        filt = {"field1": "same1"}
        count_before = orion_db.count("test_collection")
        count_filt = orion_db.count("test_collection", filt)
        # call interface
        assert orion_db.remove("test_collection", filt) == count_filt
        database = get_db(orion_db)
        assert orion_db.count("test_collection") == count_before - count_filt
        assert orion_db.count("test_collection") == 1
        loaded_config = list(database["test_collection"].find())
        assert loaded_config == test_collection[1:2]

    def test_remove_with_id(self, orion_db, test_collection):
        """Query using ``_id`` key."""
        filt = {"_id": test_collection[0]["_id"]}

        count_before = orion_db.count("test_collection")

        # call interface
        assert orion_db.remove("test_collection", filt) == 1
        database = get_db(orion_db)
        assert orion_db.count("test_collection") == count_before - 1
        loaded_configs = list(database["test_collection"].find())
        assert loaded_configs == test_collection[1:]

    def test_remove_update_indexes(self, orion_db, test_collection):
        """Verify that indexes are properly update after deletion."""
        with pytest.raises(DuplicateKeyError):
            orion_db.write("test_collection", {"_id": test_collection[0]["_id"]})
        with pytest.raises(DuplicateKeyError):
            orion_db.write("test_collection", {"_id": test_collection[1]["_id"]})

        filt = {"_id": test_collection[0]["_id"]}

        count_before = orion_db.count("test_collection")
        # call interface
        assert orion_db.remove("test_collection", filt) == 1
        assert orion_db.count("test_collection") == count_before - 1
        # Should not fail now, otherwise it means the indexes were not updated properly during
        # remove()
        orion_db.write("test_collection", filt)
        # And this should still fail
        with pytest.raises(DuplicateKeyError):
            orion_db.write("test_collection", {"_id": test_collection[1]["_id"]})


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestCount:
    """Calls :meth:`orion.core.io.database.AbstractDB.count`."""

    def test_count_default(self, orion_db, test_collection):
        """Call just with collection name."""
        found = orion_db.count("test_collection")
        assert found == len(test_collection)

    def test_count_query(self, orion_db):
        """Call with a query."""
        found = orion_db.count("test_collection", {"field1": "same1"})
        assert found == 2

    def test_count_query_with_id(self, orion_db, test_collection):
        """Call querying with unique _id."""
        found = orion_db.count("test_collection", {"_id": test_collection[2]["_id"]})
        assert found == 1

    def test_count_nothing(self, orion_db, test_collection):
        """Call with argument that will not find anything."""
        found = orion_db.count("test_collection", {"name": "lalalanotfound"})
        assert found == 0


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestIndexInformation:
    """Calls :meth:`orion.core.io.database.AbstractDB.index_information`."""

    def test_no_index(self, orion_db):
        """Test that no index is returned when there is none."""
        assert orion_db.index_information("test_collection") == {"_id_": True}

    @pytest.mark.parametrize(
        "db_test_data",
        [
            {
                "ephemeraldb_pickleddb": (False, "name", {"_id_": True}),
                "mongodb": (False, "name", {"_id_": True, "name_1": False}),
            },
            {
                "ephemeraldb_pickleddb": (True, "name", {"_id_": True, "name_1": True}),
                "mongodb": (True, "name", {"_id_": True, "name_1": True}),
            },
        ],
        indirect=True,
    )
    def test_single_index(self, orion_db, db_test_data):
        """Test that single indexes are ignored if not unique."""
        unique, key, index_information = db_test_data
        orion_db.ensure_index("experiments", [(key, Database.ASCENDING)], unique=unique)

        assert orion_db.index_information("experiments") == index_information

    @pytest.mark.parametrize(
        "db_test_data",
        [
            {
                "ephemeraldb_pickleddb": (False, "name", {"_id_": True}),
                "mongodb": (False, "name", {"_id_": True, "name_-1": False}),
            },
            {
                "ephemeraldb_pickleddb": (True, "name", {"_id_": True, "name_1": True}),
                "mongodb": (True, "name", {"_id_": True, "name_-1": True}),
            },
        ],
        indirect=True,
    )
    def test_ordered_index(self, orion_db, db_test_data):
        """Test that ordered indexes are not taken into account."""
        unique, key, index_information = db_test_data
        orion_db.ensure_index(
            "experiments", [(key, Database.DESCENDING)], unique=unique
        )

        assert orion_db.index_information("experiments") == index_information

    @pytest.mark.parametrize(
        "db_test_data",
        [
            {
                "ephemeraldb_pickleddb": (
                    False,
                    [("name", Database.DESCENDING), ("version", Database.ASCENDING)],
                    {"_id_": True},
                ),
                "mongodb": (
                    False,
                    [("name", Database.DESCENDING), ("version", Database.ASCENDING)],
                    {"_id_": True, "name_-1_version_1": False},
                ),
            },
            {
                "ephemeraldb_pickleddb": (
                    True,
                    [("name", Database.DESCENDING), ("version", Database.ASCENDING)],
                    {"_id_": True, "name_1_version_1": True},
                ),
                "mongodb": (
                    True,
                    [("name", Database.DESCENDING), ("version", Database.ASCENDING)],
                    {"_id_": True, "name_-1_version_1": True},
                ),
            },
        ],
        indirect=True,
    )
    def test_compound_index(self, orion_db, db_test_data):
        """Test representation of compound indexes."""
        unique, keys_pair, index_information = db_test_data
        orion_db.ensure_index(
            "experiments",
            keys_pair,
            unique=unique,
        )

        assert orion_db.index_information("experiments") == index_information


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestDropIndex:
    """Calls :meth:`orion.core.io.database.AbstractDB.drop_index`."""

    def test_no_index(self, orion_db):
        """Test that no index is returned when there is none."""
        with pytest.raises(DatabaseError) as exc:
            orion_db.drop_index("test_collection", "i_dont_exist")
        assert "index not found with name" in str(exc.value)

    def test_drop_single_index(self, orion_db):
        """Test with single indexes."""
        orion_db.ensure_index(
            "experiments", [("name", Database.ASCENDING)], unique=True
        )
        assert orion_db.index_information("experiments") == {
            "_id_": True,
            "name_1": True,
        }
        orion_db.drop_index("experiments", "name_1")
        assert orion_db.index_information("experiments") == {"_id_": True}

    @pytest.mark.db_types_only(["ephemeraldb", "pickleddb"])
    @pytest.mark.parametrize(
        "db_test_data",
        [
            {
                "ephemeraldb_pickleddb": (
                    [("name", Database.DESCENDING)],
                    "name_1",
                    {"_id_": True, "name_1": True},
                ),
            },
            {
                "ephemeraldb_pickleddb": (
                    [("name", Database.ASCENDING), ("version", Database.DESCENDING)],
                    "name_1_version_1",
                    {"_id_": True, "name_1_version_1": True},
                ),
            },
        ],
        indirect=True,
    )
    def test_drop_ordered_index(self, orion_db, db_test_data):
        """Test with single and compound indexes."""
        keys, stored_keys, index_information = db_test_data
        orion_db.ensure_index("experiments", keys, unique=True)
        assert orion_db.index_information("experiments") == index_information
        orion_db.drop_index("experiments", stored_keys)
        assert orion_db.index_information("experiments") == {"_id_": True}


@insert_test_collection
def test_serializable(orion_db):
    serialized = pickle.dumps(orion_db)
    deserialized = pickle.loads(serialized)
    assert len(orion_db.read("test_collection", {})) > 0
    assert orion_db.read("test_collection", {}) == deserialized.read(
        "test_collection", {}
    )
