# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysis` -- Definition of types of analysis
============================================================

.. module:: analysis
   :platform: Unix
   :synopsis: Containers for the analysers' output to the plotters
"""


class Analysis(object):
    """Base class for an analysis"""
    pass


class SingleValueAnalysis(Analysis):
    """Analysis storing a single value"""

    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value
