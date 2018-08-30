# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysers.commandline` -- Default analyser for debug purposes
=============================================================================

.. module:: commandline
   :platform: Unix
   :synopsis: Analyses the experiment by returning the best trial

"""

import logging

from orion.viz.analysers.base import BaseAnalyser
from orion.viz.analysis import SingleValueAnalysis

log = logging.getLogger(__name__)


class Commandline(BaseAnalyser):
    """This analyser will look into its :mod:`orion.core.worker.experiment.Experiment` instance
    and retrieve the best trial through the `stats` property
    """

    def __init__(self, trials, experiment):
        """Call the base class"""
        super(Commandline, self).__init__(trials, experiment)

    def analyse(self, of_type=None):
        """Find the best trial's id and return it"""
        stats = self.experiment.stats
        return SingleValueAnalysis(stats['best_trials_id'])

    @property
    def available_analysis(self):
        """Provide a single value analysis"""
        return [SingleValueAnalysis]
