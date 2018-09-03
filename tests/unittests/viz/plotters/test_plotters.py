#!/usr/bin/env python
# -*- coding= utf-8 -*-
"""Example usage and tests for mod:`orion.viz.plotters`."""
from orion.viz.analysis import SingleValueAnalysis
from orion.viz.plotters.plotterwrapper import PlotterWrapper


def test_commandline(capsys):
    """Test if the debug commandline correctly outputs the text"""
    analysis = SingleValueAnalysis(1)
    plotter = PlotterWrapper(analysis, [], {'commandline': {'debug': {}}})

    plotter.plot()

    captured = capsys.readouterr()
    assert captured.out == '1\n'
