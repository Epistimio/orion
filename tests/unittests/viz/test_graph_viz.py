#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.viz.evc.graph`."""

import pytest
import hashlib

from orion.viz.evc.graph import evc_graph
from orion.core.evc.tree import TreeNode

def md5(fname):
    '''
    Save memory md5hash
    '''
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

@pytest.fixture()
def create_graph():
    """
    Create evc_node graph
    """
    a = TreeNode("a")
    b = TreeNode("b", a)
    c = TreeNode("c", a)
    d = TreeNode("d", a)
    e = TreeNode("e", a)
    f = TreeNode("f", b)
    g = TreeNode("g", b)
    h = TreeNode("h", e)

    '''
        # Gives this tree
        # a
        # |   \  \   \
        # b    c  d   e
        # | \         |
        # f  g        h
    '''

    evc_node = a 

    '''
    a is root of EVC tree
    '''

    test_graph = evc_graph(evc_node)
    return test_graph

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
    test_graph = create_graph()
    test_graph.image_visualize('./tmp/graph.png')
    '''
    hash graph.png and compare it statically
    '''
    value = md5('./tmp/graph.png')
    assert value == '098f6bcd4621d373cade4e832627b4f6'

@pytest.fixture()
def test_create_pydot_graph():
    """
    Idem
    """
    test_graph = create_graph()
    test_graph.image_visualize_pydot('./tmp/graph.dot')
    '''
    hash graph.png and compare it statically
    '''
    value = md5('./tmp/graph.dot')
    assert value == '098f6bcd4621d373cade4e832627b4f6'

@pytest.fixture()
def test_create_tex_graph():
    """
    Idem
    """
    test_graph = create_graph()
    test_graph.tex_visualize('./tmp/graph.tex')
    '''
    hash graph.png and compare it statically
    '''
    value = md5('./tmp/graph.tex')
    assert value == '098f6bcd4621d373cade4e832627b4f6'

@pytest.fixture()
def test_create_tikz_graph():
    """
    Idem
    """
    test_graph = create_graph()
    test_graph.tikz_visualize('./tmp/graph.tikz')
    '''
    hash graph.png and compare it statically
    '''
    value = md5('./tmp/graph.tikz')
    assert value == '098f6bcd4621d373cade4e832627b4f6'
