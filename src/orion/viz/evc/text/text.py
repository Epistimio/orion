# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.evc.text` -- Build and text graph based on exp-fork tree
==============================================================================

.. module:: vizualization
   :platform: Unix
   :synopsis: Obtains graph object and outputs an text-based graph for research 
              purposes

"""
import networkx as nx 
import matplotlib.pyplot as plt

class textgraph(object):
    def __init__(self,
