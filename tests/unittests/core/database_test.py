#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.database`."""

import pytest

from orion.core.io.database import Database
from orion.core.io.database.mongodb import MongoDB
from orion.core.utils import SingletonError


@pytest.mark.usefixtures("null_db_instances")
class TestDatabaseFactory(object):
    """Test the creation of a determinate `Database` type, by a complete spefication
    of a database by-itself (this on which every `Database` acts on as part
    of its being, attributes of an `AbstractDB`) and for-itself (what essentially
    differentiates one concrete `Database` from one other).

    """

    def test_empty_first_call(self):
        """Should not be able to make first call without any arguments.

        Hegelian Ontology Primer
        ------------------------

        Type indeterminate <-> type abstracted from its property <-> No type
        """
        with pytest.raises(TypeError) as exc_info:
            Database()
        assert 'positional argument' in str(exc_info.value)

    def test_notfound_type_first_call(self):
        """Raise when supplying not implemented wrapper name."""
        with pytest.raises(NotImplementedError) as exc_info:
            Database('notfound')
        assert 'AbstractDB' in str(exc_info.value)

    def test_instatiation_and_singleton(self):
        """Test create just one object, that object persists between calls."""
        database = Database(of_type='MongoDB', name='orion_test',
                            username='user', password='pass')

        assert isinstance(database, MongoDB)
        assert database is MongoDB()
        assert database is Database()
        with pytest.raises(SingletonError) as exc_info:
            Database('fire', [], {'it_matters': 'it\'s singleton'})
        assert 'singleton' in str(exc_info.value)
