# -*- coding: utf-8 -*-
# pylint: disable=eval-used,protected-access
"""
:mod:`orion.core.cli.insert` -- Module to insert new trials
===========================================================

.. module:: insert
   :platform: Unix
   :synopsis: Insert creates new trials for a given experiment with fixed values

"""
import logging

from orion.core.cli import base as cli
from orion.core.io.evc_builder import EVCBuilder
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.worker.trial import Trial
from orion.viz.analysers.base import AnalyserWrapper
from orion.viz.plotters.plotterwrapper import PlotterWrapper

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    viz_parser = parser.add_parser('viz', help='viz help')

    orion_group = cli.get_basic_args_group(viz_parser)

    orion_group.add_argument('--analyser', type=str)
    orion_group.add_argument('--plotter', type=str)

    orion_group.set_defaults(func=main)

    return orion_group


def main(args):
    analyser_config = ExperimentBuilder().fetch_full_config(args)['analyser']
    plotter_config = ExperimentBuilder().fetch_full_config(args)['plotter']
    experiment = EVCBuilder().build_from(args)

    print(analyser_config)
    print(plotter_config)

    trials = Trial.build(experiment._db.read('trials', dict(experiment=experiment.id)))
    analyser = AnalyserWrapper(trials, experiment, analyser_config)

    plotter = PlotterWrapper(analyser.analyse(), ['png'], plotter_config)
    plotter.plot()
