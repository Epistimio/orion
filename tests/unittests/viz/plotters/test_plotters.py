#!/usr/bin/env python
# -*- coding= utf-8 -*-
"""Example usage and tests for mod:`orion.viz.plotters`."""
from orion.viz.analysis import SingleValueAnalysis
from orion.viz.plotters.plotterwrapper import PlotterWrapper

import pytest


def test_commandline(capsys):
    analysis = SingleValueAnalysis(1)
    plotter = PlotterWrapper(analysis, [], {'commandline': {'debug': {}}})

    plotter.plot()

    captured = capsys.readouterr()
    assert captured.out == '1\n'
