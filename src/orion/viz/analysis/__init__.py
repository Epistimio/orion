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

    pass


class SingleValueAnalysis(Analysis):
    """Analysis storing a single value"""

    def __init__(self, value):
        """Initialize analysis with given value"""
        self._value = value

    @property
    def value(self):
        """Return value"""
        return self._value


class CategoricalAnalysis(Analysis):
    """Analysis following this form :
    X = [categories]
    Y = [values]
    """

    def __init__(self, value):
        """Initialize analysis with given value"""
        self._value = value

    @property
    def value(self):
        """Return value"""
        return self._value

    @value.setter
    def value(self, XY_dict):
        """Change current value"""
        self._value = XY_dict
