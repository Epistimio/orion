# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysers.commandline` -- Default analyser for debug purposes
======================================================================================

.. module:: commandline
   :platform: Unix
   :synopsis: Analyses the experiment by returning the best trial

"""

import logging

from orion.viz.analysers import BaseAnalyser
from orion.viz.analysis import SingleValueAnalysis

log = logging.getLogger(__name__)


class Commandline(BaseAnalyser):
    """Return best trial"""

    def __init__(self, trials, experiment):
        super(Commandline, self).__init__(trials, experiment)

    def analyse(self, of_type=None):
        stats = self.experiment.stats
        return SingleValueAnalysis(stats['best_trials_id'])

    @property
    def available_analysis(self):
        return [SingleValueAnalysis]
