#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.core.io.database`."""

import pytest

from orion.core.io.database import ReadOnlyDB, database_factory
from orion.storage.base import setup_storage


@pytest.mark.usefixtures("null_db_instances")
class TestDatabaseFactory:
    """Test the creation of a determinate `Database` type, by a complete specification
    of a database by-itself (this on which every `Database` acts on as part
    of its being, attributes of an `Database`) and for-itself (what essentially
    differentiates one concrete `Database` from one other).

    """

    def test_notfound_type_first_call(self):
        """Raise when supplying not implemented wrapper name."""
        with pytest.raises(NotImplementedError) as exc_info:
            database_factory.create("notfound")

        assert "Database" in str(exc_info.value)


@pytest.mark.usefixtures("null_db_instances")
class TestReadOnlyDatabase:
    """Test coherence of read-only database and its wrapped database."""

    def test_valid_attributes(self, storage):
        """Test attributes are coherent from view and wrapped database."""
        database = storage._db
        readonly_database = ReadOnlyDB(database)

        assert readonly_database.host == database.host
        assert readonly_database.port == database.port

    def test_read(self, hacked_exp):
        """Test read is coherent from view and wrapped database."""
        database = setup_storage()._db
        readonly_database = ReadOnlyDB(database)

        args = {
            "collection_name": "experiments",
            "query": {"name": "supernaedo2-dendi"},
        }
        readonly_result = readonly_database.read(**args)
        result = database.read(**args)

        assert len(result) > 0  # Otherwise the test is pointless
        assert readonly_result == result

    def test_invalid_attributes(self, storage):
        """Test that attributes for writing are not accessible."""
        database = storage._db
        readonly_database = ReadOnlyDB(database)

        # Test that database.ensure_index indeed exists
        database.ensure_index
        with pytest.raises(AttributeError):
            readonly_database.ensure_index

        # Test that database.write indeed exists
        database.write
        with pytest.raises(AttributeError):
            readonly_database.write

        # Test that database.read_and_write indeed exists
        database.read_and_write
        with pytest.raises(AttributeError):
            readonly_database.read_and_write

        # Test that database.remove indeed exists
        database.remove
        with pytest.raises(AttributeError):
            readonly_database.remove
