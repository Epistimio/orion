#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.io.database.pickleddb`."""
import pytest
from test_database import clean_db, orion_db

from orion.core.io.database import Database
from orion.core.io.database.ephemeraldb import (
    EphemeralCollection,
    EphemeralDB,
    EphemeralDocument,
)


@pytest.fixture(scope="module", autouse=True)
def db_type(pytestconfig, request):
    """Return the string identifier of a EphemeralDB if the --mongodb option is
    not active"""
    if pytestconfig.getoption("--mongodb"):
        pytest.skip("ephemeraldb tests disabled")
    yield "ephemeraldb"


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


# TESTS SET


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
            [("hello", Database.DESCENDING), ("idontexist", Database.ASCENDING)],
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
