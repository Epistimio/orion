#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test base functionalities of :mod:`orion.core.utils`."""

import pytest

from orion.core.utils import Factory


def test_factory_subclasses_detection():
    """Verify that meta-class Factory finds all subclasses"""
    class Base(object):
        pass

    class A(Base):
        pass

    class B(Base):
        pass

    class AA(A):
        pass

    class AB(A):
        pass

    class AAA(AA):
        pass

    class AA_AB(AA, AB):
        pass

    class MyFactory(Base, metaclass=Factory):
        pass

    assert type(MyFactory(of_type="A")) is A
    assert type(MyFactory(of_type="B")) is B
    assert type(MyFactory(of_type="AA")) is AA
    assert type(MyFactory(of_type="AAA")) is AAA
    assert type(MyFactory(of_type="AA_AB")) is AA_AB

    # Test if there is duplicates
    assert set(MyFactory.types) == set((A, B, AA, AB, AAA, AA_AB))
    assert len(MyFactory.types) == len(set(MyFactory.types))

    with pytest.raises(NotImplementedError) as exc_info:
        MyFactory(of_type="random")
    assert "Could not find implementation of Base, type = 'random'" in str(exc_info.value)
