# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysis` -- Definition of types of analysis
============================================================

.. module:: analysis
   :platform: Unix
   :synopsis: Containers for the analysers' output to the plotters
"""

from abc import ABCMeta


class Analysis(object, metaclass=ABCMeta):
    """Base class for an analysis"""

    @property
    def value(self):
        return None


class SingleValueAnalysis(Analysis):
    """Analysis storing a single value"""
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


class TimeSeriesAnalysis(Analysis):
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, XY_dict):
        self._value = XY_dict
