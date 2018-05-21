#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.viz.evc.text`."""


import pytest

from orion.viz.evc.text import text_vizualize

@pytest.fixture()
def create_text_graph():
    pass

@pytest.fixture()
def test_create_text_graph():
    """
    so I thought of using the creation of the graphs for tests.
    Given two identical graphs and given the similar procedure
    that the graphs are generated. The MD5 hash of the graphs
    woud be similar to each other. In this case we would concatenate
    the whole text based graph into a long string and then hash it
    the result of the MD5 hash should be identical to the one
    created thus a test assertion would be possible.
    """
    pass
