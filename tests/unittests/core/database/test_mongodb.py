#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.io.database.pickleddb`."""
import functools
from timeit import timeit

import pymongo
import pytest
from pymongo import MongoClient

from orion.core.io.database import Database, DatabaseError, DuplicateKeyError
from orion.core.io.database.mongodb import AUTH_FAILED_MESSAGES, MongoDB

from .conftest import insert_test_collection
from .test_database import get_db


@pytest.fixture(scope="module", autouse=True)
def db_type(pytestconfig, request):
    """Return the string identifier of a MongoDB if the --mongodb option is
    active"""
    if not pytestconfig.getoption("--mongodb"):
        pytest.skip("mongodb tests disabled")
    yield "mongodb"


@pytest.fixture()
def patch_mongo_client(monkeypatch, db_type):
    """Patch ``pymongo.MongoClient`` to force serverSelectionTimeoutMS to 1."""
    if db_type != "mongodb":
        pytest.skip("mongodb test only")

    def mock_class(*args, **kwargs):
        # 1 sec, defaults to 20 secs otherwise
        kwargs["serverSelectionTimeoutMS"] = 1.0
        # NOTE: Can't use pymongo.MongoClient otherwise there is an infinite
        # recursion; mock(mock(mock(mock(...(MongoClient)...))))
        return MongoClient(*args, **kwargs)

    monkeypatch.setattr("pymongo.MongoClient", mock_class)


# TESTS SET


@pytest.mark.usefixtures("clean_db")
@pytest.mark.drop_collections(["new_collection"])
class TestEnsureIndex:
    """Calls to :meth:`orion.core.io.database.AbstractDB.ensure_index`."""

    def test_unique_index(self, orion_db):
        """Index should be set as unique in mongo database's index information."""
        assert (
            "name_1_metadata.user_1"
            not in get_db(orion_db)["experiments"].index_information()
        )
        orion_db.ensure_index(
            "experiments",
            [("name", Database.ASCENDING), ("metadata.user", Database.ASCENDING)],
            unique=True,
        )
        index_information = get_db(orion_db)["experiments"].index_information()
        assert "name_1_metadata.user_1" in index_information
        assert index_information["name_1_metadata.user_1"]["unique"]


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestDropIndex:
    """Calls :meth:`orion.core.io.database.AbstractDB.drop_index`."""

    @pytest.mark.parametrize(
        "db_test_data",
        [
            {
                "mongodb": (
                    [("name", Database.ASCENDING)],
                    [("name", Database.DESCENDING)],
                    "name_1",
                    "name_-1",
                    {"_id_": True, "name_1": False, "name_-1": False},
                    {"_id_": True, "name_-1": False},
                ),
            },
            {
                "mongodb": (
                    [("name", Database.ASCENDING), ("version", Database.DESCENDING)],
                    [("name", Database.DESCENDING), ("version", Database.ASCENDING)],
                    "name_1_version_-1",
                    "name_-1_version_1",
                    {
                        "_id_": True,
                        "name_1_version_-1": False,
                        "name_-1_version_1": False,
                    },
                    {"_id_": True, "name_-1_version_1": False},
                ),
            },
        ],
        indirect=True,
    )
    def test_drop_ordered_index(self, orion_db, db_test_data):
        """Test with single and compound indexes."""
        (
            keys_1,
            keys_2,
            stored_keys_1,
            stored_keys_2,
            index_information_initial,
            index_information_wo_1,
        ) = db_test_data
        orion_db.ensure_index("experiments", keys_1)
        orion_db.ensure_index("experiments", keys_2)
        assert orion_db.index_information("experiments") == index_information_initial
        orion_db.drop_index("experiments", stored_keys_1)
        assert orion_db.index_information("experiments") == index_information_wo_1
        with pytest.raises(DatabaseError) as exc:
            orion_db.drop_index("experiments", stored_keys_1)
        assert "index not found with name" in str(exc.value)
        orion_db.drop_index("experiments", stored_keys_2)
        assert orion_db.index_information("experiments") == {"_id_": True}

    def test_drop_unique_index(self, orion_db):
        """Test with single indexes."""
        orion_db.ensure_index("hello", [("bonjour", Database.DESCENDING)], unique=True)
        index_info = orion_db.index_information("hello")
        assert index_info == {"_id_": True, "bonjour_-1": True}
        orion_db.drop_index("hello", "bonjour_-1")
        index_info = orion_db.index_information("hello")
        assert index_info == {"_id_": True}


# MONGODB ONLY TESTS


@pytest.mark.usefixtures("null_db_instances")
class TestConnection:
    """Create a :class:`orion.core.io.database.mongodb.MongoDB`, check connection cases."""

    @pytest.mark.usefixtures("patch_mongo_client")
    def test_bad_connection(self, monkeypatch):
        """Raise when connection cannot be achieved."""
        monkeypatch.setattr(
            MongoDB, "initiate_connection", MongoDB.initiate_connection.__wrapped__
        )
        with pytest.raises(pymongo.errors.ConnectionFailure) as exc_info:
            MongoDB(
                host="asdfada",
                port=123,
                name="orion",
                username="uasdfaf",
                password="paasdfss",
            )

        monkeypatch.undo()

        # Verify that the wrapper converts it properly to DatabaseError
        with pytest.raises(DatabaseError) as exc_info:
            MongoDB(
                host="asdfada",
                port=123,
                name="orion",
                username="uasdfaf",
                password="paasdfss",
            )
        assert "Connection" in str(exc_info.value)

    def test_bad_authentication(self, monkeypatch):
        """Raise when authentication cannot be achieved."""
        monkeypatch.setattr(
            MongoDB, "initiate_connection", MongoDB.initiate_connection.__wrapped__
        )
        with pytest.raises(pymongo.errors.OperationFailure) as exc_info:
            MongoDB(name="orion_test", username="uasdfaf", password="paasdfss")
        assert any(m in str(exc_info.value) for m in AUTH_FAILED_MESSAGES)

        monkeypatch.undo()

        with pytest.raises(DatabaseError) as exc_info:
            MongoDB(name="orion_test", username="uasdfaf", password="paasdfss")
        assert "Authentication" in str(exc_info.value)

    def test_connection_with_uri(self):
        """Check the case when connecting with ready `uri`."""
        orion_db = MongoDB("mongodb://user:pass@localhost/orion_test")
        assert orion_db.host == "localhost"
        assert orion_db.port == 27017
        assert orion_db.username == "user"
        assert orion_db.password == "pass"
        assert orion_db.name == "orion_test"

    def test_overwrite_uri(self):
        """Check the case when connecting with ready `uri`."""
        orion_db = MongoDB(
            "mongodb://user:pass@localhost:27017/orion_test",
            port=1231,
            name="orion",
            username="lala",
            password="pass",
        )
        assert orion_db.host == "localhost"
        assert orion_db.port == 27017
        assert orion_db.username == "user"
        assert orion_db.password == "pass"
        assert orion_db.name == "orion_test"

    def test_overwrite_partial_uri(self, monkeypatch):
        """Check the case when connecting with partial `uri`."""
        monkeypatch.setattr(MongoDB, "initiate_connection", lambda self: None)

        orion_db = MongoDB(
            "mongodb://localhost",
            port=1231,
            name="orion",
            username="lala",
            password="none",
        )
        orion_db._sanitize_attrs()
        assert orion_db.host == "localhost"
        assert orion_db.port == 1231
        assert orion_db.username == "lala"
        assert orion_db.password == "none"
        assert orion_db.name == "orion"

    def test_change_server_timeout(self):
        """Test that the server timeout is correctly changed."""
        assert (
            timeit(
                lambda: MongoDB(
                    username="user",
                    password="pass",
                    name="orion_test",
                    serverSelectionTimeoutMS=1000,
                ),
                number=1,
            )
            <= 2
        )


@pytest.mark.usefixtures("clean_db")
@insert_test_collection
class TestExceptionWrapper:
    """Call to methods wrapped with `mongodb_exception_wrapper()`."""

    def test_duplicate_key_error(self, monkeypatch, orion_db, test_collection):
        """Should raise generic DuplicateKeyError."""
        # Add unique indexes to force trigger of DuplicateKeyError on write()
        orion_db.ensure_index(
            "test_collection",
            [("field0", Database.ASCENDING), ("datetime", Database.ASCENDING)],
            unique=True,
        )

        config_to_add = test_collection[0]
        config_to_add.pop("_id")

        query = {"_id": test_collection[1]["_id"]}

        # Make sure it raises pymongo.errors.DuplicateKeyError when there is no
        # wrapper
        monkeypatch.setattr(
            orion_db,
            "read_and_write",
            functools.partial(orion_db.read_and_write.__wrapped__, orion_db),
        )
        with pytest.raises(pymongo.errors.DuplicateKeyError) as exc_info:
            orion_db.read_and_write("test_collection", query, config_to_add)

        monkeypatch.undo()

        # Verify that the wrapper converts it properly to DuplicateKeyError
        with pytest.raises(DuplicateKeyError) as exc_info:
            orion_db.read_and_write("test_collection", query, config_to_add)
        assert "duplicate key error" in str(exc_info.value)

    def test_bulk_duplicate_key_error(self, monkeypatch, orion_db, test_collection):
        """Should raise generic DuplicateKeyError."""
        # Make sure it raises pymongo.errors.BulkWriteError when there is no
        # wrapper
        monkeypatch.setattr(
            orion_db, "write", functools.partial(orion_db.write.__wrapped__, orion_db)
        )
        with pytest.raises(pymongo.errors.BulkWriteError) as exc_info:
            orion_db.write("test_collection", test_collection)

        monkeypatch.undo()

        # Verify that the wrapper converts it properly to DuplicateKeyError
        with pytest.raises(DuplicateKeyError) as exc_info:
            orion_db.write("test_collection", test_collection)
        assert "duplicate key error" in str(exc_info.value)

    def test_non_converted_errors(self, orion_db, test_collection):
        """Should raise OperationFailure.

        This is because _id inside exp_config[0][0] cannot be set. It is an
        immutable key of the collection.

        """
        config_to_add = test_collection[0]

        query = {"_id": test_collection[1]["_id"]}

        with pytest.raises(pymongo.errors.OperationFailure):
            orion_db.read_and_write("test_collection", query, config_to_add)


def test_repr(orion_db: MongoDB):
    assert str(orion_db) == (
        f"MongoDB(host=localhost, name=orion_test, port=27017, username=user, "
        f"password=pass, options={orion_db.options})"
    )
