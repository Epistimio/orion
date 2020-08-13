#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.serving.plots_resources` -- Module responsible for the plots/ REST endpoint
========================================================================================

.. module:: plots_resources
   :platform: Unix
   :synopsis: Serves all the requests made to plots/ REST endpoint
"""
from falcon import Request, Response

from orion.client import ExperimentClient
from orion.serving.parameters import retrieve_experiment
from orion.storage.base import get_storage


class PlotsResource(object):
    """Serves all the requests made to plots/ REST endpoint"""

    def __init__(self):
        self.storage = get_storage()

    def on_get_regret(self, req: Request, resp: Response, experiment_name: str):
        """
        Handle GET requests for plotting regret plots on plots/regret/:experiment
        where ``experiment`` is the user-defined name of the experiment.
        """
        experiment = ExperimentClient(retrieve_experiment(experiment_name), None)
        resp.body = experiment.plot.regret().to_json()
