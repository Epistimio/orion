"""Example file for pytests."""

import metaopt


def test_tests():
    """Dummy test function."""
    print(metaopt.__version__)
    assert True


def test_dict_equal():
    """Test how pytest works for dicts."""
    a = {'a': 2, 'b': [1, 2], 'c': {'a': 5, 'b': 6}}
    b = {'a': 2, 'b': [1, 2], 'd': {'a': 5, 'b': 6}}
    c = {'a': 2, 'b': [1, 2], 'c': {'a': 5, 'b': 6}}
    assert a == c
    assert a != b
