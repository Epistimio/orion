#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.utils.Concept`."""
import pytest

from orion.concepts.base import BaseConceptWrapper
from orion.concepts.baseconceptimpl import BaseConceptImpl
from orion.concepts.implwrapper import ImplWrapper
from orion.concepts.implwrapper.otherconceptimpl import OtherConceptImpl
from orion.concepts.subconcept import SubConcept


@pytest.fixture
def string_config():
    """Return a config with only a string"""
    return {"concept": "BaseConceptImpl"}


@pytest.fixture
def dict_config_no_args():
    """Return a config as a dictionary without arguments"""
    return {"concept": {"BaseConceptImpl": {}}}


@pytest.fixture
def dict_config_args():
    """Return a config as a dictionary with an argument"""
    return {"concept": {"BaseConceptImpl": {"arg": 2}}}


@pytest.fixture
def depth_2_config():
    """Return a config of cascading types with an argument"""
    return {"concept": {"SubConcept": {"BaseConceptImpl": {"arg": 2}}}}


@pytest.fixture
def impl_submodule_config():
    """Return a config of a submodule type"""
    return {"concept": {"ImplWrapper": {"OtherConceptImpl": {"arg": 2}}}}


def test_string_init(string_config):
    """Check if initialization works with just the string name"""
    concept = BaseConceptWrapper(**string_config)

    assert type(concept.concept) is BaseConceptImpl
    assert concept.concept.arg == 0


def test_init_without_args(dict_config_no_args):
    """Check if initialization works with a dictionary"""
    concept = BaseConceptWrapper(**dict_config_no_args)

    assert type(concept.concept) is BaseConceptImpl
    assert concept.concept.arg == 0


def test_init_with_args(dict_config_args):
    """Check if initialization works with a dictionary and args"""
    concept = BaseConceptWrapper(**dict_config_args)

    assert type(concept.concept) is BaseConceptImpl
    assert concept.concept.arg == 2


def test_depth_2_config(depth_2_config):
    """Check if cascading initialization work"""
    concept = BaseConceptWrapper(**depth_2_config)

    assert type(concept.concept) is SubConcept


def test_impl_in_submodule(impl_submodule_config):
    """Check if initialization of submodules work"""
    concept = BaseConceptWrapper(**impl_submodule_config)

    assert type(concept.concept) is ImplWrapper
    assert type(concept.concept.concept) is OtherConceptImpl
