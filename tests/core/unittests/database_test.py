#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`metaopt.io.database`."""

import pytest

from metaopt.io.database import (Database, MongoDB)


def test_empty_first_call():
    """Should not be able to make first call without any arguments.

    Hegelian Ontology Primer
    ------------------------

    Type indeterminate <-> type abstracted from its property <-> No type
    """
    with pytest.raises(TypeError) as exc_info:
        Database()
    assert 'abstract class' in str(exc_info.value)


def test_notfound_type_first_call():
    """Raise when supplying not implemented wrapper name."""
    with pytest.raises(NotImplementedError) as exc_info:
        Database('notfound')
    assert 'AbstractDB' in str(exc_info.value)


def test_instatiation_and_singleton():
    """Test create just one object, that object persists between calls."""
    database = Database(dbtype='MongoDB', dbname='metaopt_test',
                        username='user', password='pass')

    assert Database() is database
    assert Database('fire', [], {'doesnt_matter': 'it\'s singleton'}) is database
    MongoDB.instance = None  # Set singular instance to None for independent tests
