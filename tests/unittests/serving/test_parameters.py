"""Performs unit tests for orion.serving.parameters.py"""
import pytest

from orion.serving.parameters import verify_parameter_type


def test_bool_parameter():
    """Tests scenarios for 'bool' parameters"""
    tested_type = 'bool'

    assert verify_parameter_type('true', tested_type)
    assert verify_parameter_type('True', tested_type)
    assert verify_parameter_type('false', tested_type)
    assert verify_parameter_type('False', tested_type)

    assert not verify_parameter_type('', tested_type)
    assert not verify_parameter_type('0', tested_type)
    assert not verify_parameter_type('42', tested_type)
    assert not verify_parameter_type('42.0', tested_type)
    assert not verify_parameter_type('a', tested_type)
    assert not verify_parameter_type('nan', tested_type)
    assert not verify_parameter_type('None', tested_type)

    with pytest.raises(AttributeError):
        verify_parameter_type(None, tested_type)


def test_int_parameter():
    """Tests scenarios for 'int' parameters"""
    tested_type = 'int'

    assert verify_parameter_type('0', tested_type)
    assert verify_parameter_type('42', tested_type)
    assert verify_parameter_type('-42', tested_type)

    assert not verify_parameter_type('42.0', tested_type)
    assert not verify_parameter_type('true', tested_type)
    assert not verify_parameter_type('false', tested_type)
    assert not verify_parameter_type('', tested_type)
    assert not verify_parameter_type('a', tested_type)
    assert not verify_parameter_type('nan', tested_type)
    assert not verify_parameter_type('None', tested_type)


def test_str_parameter():
    """Tests scenarios for 'str' parameters"""
    tested_type = 'str'

    assert verify_parameter_type('0', tested_type)
    assert verify_parameter_type('42', tested_type)
    assert verify_parameter_type('-42', tested_type)
    assert verify_parameter_type('42.0', tested_type)
    assert verify_parameter_type('true', tested_type)
    assert verify_parameter_type('false', tested_type)
    assert verify_parameter_type('', tested_type)
    assert verify_parameter_type('a', tested_type)
    assert verify_parameter_type('nan', tested_type)
    assert verify_parameter_type('None', tested_type)


def test_unknown_parameter():
    """Tests scenarios for unknown parameters"""
    tested_type = 'unknown'

    assert not verify_parameter_type('0', tested_type)
    assert not verify_parameter_type('42', tested_type)
    assert not verify_parameter_type('-42', tested_type)
    assert not verify_parameter_type('42.0', tested_type)
    assert not verify_parameter_type('true', tested_type)
    assert not verify_parameter_type('false', tested_type)
    assert not verify_parameter_type('', tested_type)
    assert not verify_parameter_type('a', tested_type)
    assert not verify_parameter_type('nan', tested_type)
    assert not verify_parameter_type('None', tested_type)
