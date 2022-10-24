#!/usr/bin/env python
"""Collection of fixtures used in the database tests."""
import os
from datetime import datetime

import pytest

from orion.core.io.database import Database
from orion.core.io.database.ephemeraldb import EphemeralDB
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB

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


@pytest.fixture(scope="module", autouse=True, params=_DB_TYPES)
def db_type(pytestconfig, request):
    """Return the string identifier of a supported database type based on the
    --mongodb option

    If `--mongodb` is active, only MongoDB tests will be run. Otherwise,
    all non-MongoDB will be run.
    """
    if request.param == "mongodb" and not pytestconfig.getoption("--mongodb"):
        pytest.skip(f"{request.param} tests disabled")
    elif request.param != "mongodb" and pytestconfig.getoption("--mongodb"):
        pytest.skip(f"{request.param} tests disabled")
    yield request.param


@pytest.fixture(scope="module")
def orion_db(db_type):
    """Return a supported database wrapper instance initiated with test opts."""
    if db_type == "ephemeraldb":
        EphemeralDB.instance = None
        orion_db = EphemeralDB()
    elif db_type == "mongodb":
        MongoDB.instance = None
        orion_db = MongoDB(username="user", password="pass", name="orion_test")
    elif db_type == "pickleddb":
        PickledDB.instance = None
        orion_db = PickledDB(host="orion_db.pkl")
    else:
        raise ValueError("Invalid database type")

    yield orion_db

    if db_type == "ephemeraldb":
        pass
    elif db_type == "mongodb":
        orion_db.close_connection()
    elif db_type == "pickleddb":
        pass


@pytest.fixture()
def clean_db(orion_db):
    """Cleaned the current database prior a test."""
    if isinstance(orion_db, EphemeralDB):
        pass
    elif isinstance(orion_db, MongoDB):
        pass
    elif isinstance(orion_db, PickledDB):
        if os.path.exists(orion_db.host):
            os.remove(orion_db.host)

    database = get_db(orion_db)

    # Drop experiment data
    database["experiments"].drop()
    database["lying_trials"].drop()
    database["trials"].drop()
    database["workers"].drop()
    database["resources"].drop()

    dump_db(orion_db, database)

    yield database


@pytest.fixture()
def db_test_data(request, db_type):
    """Return test data corresponding to the current database type"""
    for key in request.param:
        if db_type in key:
            yield request.param[key]
            break
    else:
        raise ValueError("Invalid database type")


@pytest.fixture(autouse=True)
def insert_collections(request, orion_db, clean_db):
    """Drop a collection prior a test"""
    collections_data = (
        request.node.get_closest_marker("insert_collections").args[0]
        if request.node.get_closest_marker("insert_collections")
        else {}
    )
    for name, data in collections_data.items():
        clean_db[name].drop()
        clean_db[name].insert_many(data)

    dump_db(orion_db, clean_db)

    yield collections_data

    for name in collections_data.keys():
        clean_db[name].drop()

    dump_db(orion_db, clean_db)


@pytest.fixture(autouse=True)
def drop_collections(request, orion_db):
    """Drop a collection prior a test"""
    db = get_db(orion_db)
    collections = (
        request.node.get_closest_marker("drop_collections").args[0]
        if request.node.get_closest_marker("drop_collections")
        else tuple()
    )
    for collection in collections:
        db[collection].drop()
    yield
    for collection in collections:
        db[collection].drop()


@pytest.fixture(autouse=True)
def skip_if_not_db_type(request, db_type):
    """Skip test if th database type does no match the database type marker"""
    db_types_only = request.node.get_closest_marker("db_types_only")
    if db_types_only and db_type not in db_types_only.args[0]:
        pytest.skip(f"{db_types_only.args[0]} test only")


insert_test_collection = pytest.mark.insert_collections(
    {
        "test_collection": [
            {
                "_id": 0,
                "field0": "same0",
                "field1": "same1",
                "datetime": datetime(2017, 11, 22, 0, 0, 0),
                "same_field": "same",
                "unique_field": "unique0",
                "same_comp": {"ound": "same_compound"},
                "unique_comp": {"ound": "compound0"},
            },
            {
                "_id": 1,
                "field0": "same0",
                "field1": "diff1",
                "datetime": datetime(2017, 11, 23, 0, 0, 0),
                "same_field": "same",
                "unique_field": "unique1",
                "same_comp": {"ound": "same_compound"},
                "unique_comp": {"ound": "compound1"},
            },
            {
                "_id": 2,
                "field0": "diff0",
                "field1": "same1",
                "datetime": datetime(2017, 11, 24, 0, 0, 0),
                "same_field": "same",
                "unique_field": "unique2",
                "same_comp": {"ound": "same_compound"},
                "unique_comp": {"ound": "compound2"},
            },
        ]
    }
)


@pytest.fixture()
def test_collection(insert_collections):
    """Drop a collection prior a test"""
    yield insert_collections["test_collection"]


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
