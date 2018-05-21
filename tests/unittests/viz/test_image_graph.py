#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.viz.evc.graph`."""

import pytest

from orion.viz.evc.graph import image_vizualize

@pytest.fixture()
def create_image_graph():
    """
    Create an image graph
    """
    pass

@pytest.fixture()
def test_create_image_graph():
    """
    Given two identical graphs and given the similar procedure
    that the graphs are generated. The MD5 hash of the graphs
    woud be similar to each other. In this case we would load
    the graph into memory, hash the whole image into a MD5
    string. This assertion would logically be the correct
    graph generated to the one we would want.
    """
    pass
