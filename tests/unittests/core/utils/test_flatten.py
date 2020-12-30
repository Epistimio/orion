"""Example usage and tests for :mod:`orion.core.utils.flatten`."""
from orion.core.utils.flatten import flatten, unflatten


def test_basic():
    """Test basic functionality of flatten"""
    d = {"a": {"b": 2, "c": 3}, "c": {"d": 3, "e": 4}}
    assert flatten(d) == {"a.b": 2, "a.c": 3, "c.d": 3, "c.e": 4}


def test_handle_double_ref():
    """Test proper handling of double references in dicts"""
    a = {"b": 2, "c": 3}
    d = {"a": a, "d": {"e": a}}
    assert flatten(d) == {"a.b": 2, "a.c": 3, "d.e.b": 2, "d.e.c": 3}


def test_unflatten():
    """Test than unflatten(flatten(x)) is idempotent"""
    a = {"b": 2, "c": 3}
    d = {"a": a, "d": {"e": a}}
    assert unflatten(flatten(d)) == d
