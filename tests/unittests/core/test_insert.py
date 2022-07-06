#!/usr/bin/env python
"""Perform unit-test on functions user for insert cli."""
import numbers

import pytest

from orion.core.cli.insert import _validate_input_value
from orion.core.io.space_builder import SpaceBuilder


@pytest.fixture()
def real_space():
    """Fixture for real space"""
    return SpaceBuilder().build({"x": "uniform(-10,20)"})


@pytest.fixture()
def integer_space():
    """Fixture for integer space"""
    return SpaceBuilder().build({"x": "uniform(-10,20,discrete=True)"})


@pytest.fixture()
def categorical_space():
    """Fixture for categorical space"""
    return SpaceBuilder().build({"x": "choices([10.1,11,'12','string'])"})


def test_validate_input_value_real_real(real_space):
    """Test if real value passed to real space is validated properly"""
    namespace = "x"
    is_valid, casted_value = _validate_input_value("10.0", real_space, namespace)
    assert is_valid
    assert isinstance(casted_value, numbers.Number)


def test_validate_input_value_real_integer(real_space):
    """Test if integer value passed to real space is validated properly"""
    namespace = "x"
    is_valid, casted_value = _validate_input_value("10", real_space, namespace)
    assert is_valid
    assert isinstance(casted_value, numbers.Number)


def test_validate_input_value_real_string(real_space):
    """Test if string value passed to real space is rejected properly"""
    namespace = "x"
    is_valid, casted_value = _validate_input_value("string", real_space, namespace)
    assert not is_valid


def test_validate_input_value_real_out_of_bound(real_space):
    """Test if out of bound values passed to real space are rejected properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("100.0", real_space, namespace)
    assert not is_valid

    is_valid, casted_value = _validate_input_value("100", real_space, namespace)
    assert not is_valid


def test_validate_input_value_integer_real(integer_space):
    """Test if real value passed to integer space is validated properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("10.0", integer_space, namespace)
    assert is_valid
    assert isinstance(casted_value, numbers.Number)


def test_validate_input_value_integer_integer(integer_space):
    """Test if integer value passed to integer space is validated properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("10", integer_space, namespace)
    assert is_valid
    assert isinstance(casted_value, numbers.Number)


def test_validate_input_value_integer_string(integer_space):
    """Test if string value passed to integer space is rejected properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("string", integer_space, namespace)
    assert not is_valid


def test_validate_input_value_integer_out_of_bound(integer_space):
    """Test if out of bound values passed to integer space are rejected properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("100.0", integer_space, namespace)
    assert not is_valid

    is_valid, casted_value = _validate_input_value("100", integer_space, namespace)
    assert not is_valid


def test_validate_input_value_categorical_real_hit(categorical_space):
    """Test if real value passed to categorical space is validated properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("10.1", categorical_space, namespace)
    assert is_valid
    assert isinstance(casted_value, numbers.Number)


def test_validate_input_value_categorical_real_nohit(categorical_space):
    """Test if bad real value passed to categorical space is rejected properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("10", categorical_space, namespace)
    assert not is_valid

    is_valid, casted_value = _validate_input_value("10.0", categorical_space, namespace)
    assert not is_valid

    is_valid, casted_value = _validate_input_value("10.2", categorical_space, namespace)
    assert not is_valid


def test_validate_input_value_categorical_integer_hit(categorical_space):
    """Test if integer value passed to categorical space is validated properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("11", categorical_space, namespace)
    assert is_valid
    assert isinstance(casted_value, numbers.Number)

    is_valid, casted_value = _validate_input_value("11.0", categorical_space, namespace)
    assert is_valid
    assert isinstance(casted_value, numbers.Number)


def test_validate_input_value_categorical_integer_nohit(categorical_space):
    """Test if bad integer value passed to categorical space is rejected properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value("15", categorical_space, namespace)
    assert not is_valid


def test_validate_input_value_categorical_string_number(categorical_space):
    """Test if string number value passed to categorical space is validated properly"""
    namespace = "x"

    # Make sure integer 12 does not pass
    is_valid, casted_value = _validate_input_value("12", categorical_space, namespace)
    assert not is_valid

    # Now test "12" as a string
    is_valid, casted_value = _validate_input_value("'12'", categorical_space, namespace)
    assert is_valid
    assert isinstance(casted_value, str)


def test_validate_input_value_categorical_string_value(categorical_space):
    """Test if literal string value passed to categorical space is validated properly"""
    namespace = "x"

    is_valid, casted_value = _validate_input_value(
        "random", categorical_space, namespace
    )
    assert not is_valid

    is_valid, casted_value = _validate_input_value(
        "string", categorical_space, namespace
    )
    assert is_valid
    assert isinstance(casted_value, str)
