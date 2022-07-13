#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.io.database.pickleddb`."""
import logging
import random
import uuid
from multiprocessing import Pool

import pytest
from filelock import FileLock, SoftFileLock, Timeout

from orion.core.io.database import DatabaseTimeout, DuplicateKeyError
from orion.core.io.database.ephemeraldb import EphemeralCollection
from orion.core.io.database.pickleddb import (
    PickledDB,
    _create_lock,
    find_unpickable_doc,
    find_unpickable_field,
    local_file_systems,
)

from .test_database import get_db


@pytest.fixture(scope="module", autouse=True)
def db_type(pytestconfig, request):
    """Return the string identifier of a PickledDB if the --mongodb option is
    not active"""
    if pytestconfig.getoption("--mongodb"):
        pytest.skip("pickleddb tests disabled")
    yield "pickleddb"


@pytest.mark.usefixtures("clean_db")
class TestReadAndWrite:
    """Calls to :meth:`orion.core.io.database.AbstractDB.read_and_write`."""

    def test_logging_when_getting_file_lock(self, caplog, orion_db):
        """When logging.level is ERROR, there should be no logging."""
        logging.basicConfig(level=logging.INFO)
        caplog.clear()
        caplog.set_level(logging.ERROR)
        # any operation will trigger the lock.
        orion_db.read("experiments", {"name": "supernaedo2", "metadata.user": "dendi"})

        assert "acquired on orion_db.pkl.lock" not in caplog.text


# TESTS SET


def write(field, i):
    """Write the given value to the pickled db."""
    PickledDB.instance = None
    orion_db = PickledDB(host="orion_db.pkl")
    try:
        orion_db.write("concurrent", {field: i})
    except DuplicateKeyError:
        print("dup")

    print(field, i)


@pytest.mark.usefixtures("clean_db")
class TestConcurreny:
    """Test concurrent operations"""

    def test_concurrent_writes(self, orion_db):
        """Test that concurrent writes all get written properly"""
        orion_db.ensure_index("concurrent", "diff")

        assert orion_db.count("concurrent", {"diff": {"$gt": -1}}) == 0

        with Pool(10) as pool:
            pool.starmap(write, (("diff", i) for i in range(10)))

        assert orion_db.count("concurrent", {"diff": {"$gt": -1}}) == 10

    def test_concurrent_unique_writes(self, orion_db):
        """Test that concurrent writes cannot duplicate unique fields"""
        orion_db.ensure_index("concurrent", "unique", unique=True)

        assert orion_db.count("concurrent", {"unique": 1}) == 0

        with Pool(10) as pool:
            pool.starmap(write, (("unique", 1) for i in range(10)))

        assert orion_db.count("concurrent", {"unique": 1}) == 1


def test_empty_file(orion_db):
    """Check that db loading can handle empty files"""
    with open(orion_db.host, "wb") as f:
        f.write(b"")
    get_db(orion_db)


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


def test_query_timeout(monkeypatch, orion_db):
    """Verify that filelock.Timeout is caught and reraised as DatabaseTimeout"""
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


def test_repr(orion_db: PickledDB):
    assert (
        str(orion_db) == f"PickledDB(host={orion_db.host}, timeout={orion_db.timeout})"
    )
