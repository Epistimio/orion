#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module responsible for the plots/ REST endpoint
================================================

Serves all the requests made to plots/ REST endpoint.

"""
from falcon import Request, Response

from orion.client import ExperimentClient
from orion.serving.parameters import retrieve_experiment
from orion.storage.base import get_storage


class PlotsResource(object):
    """Serves all the requests made to plots/ REST endpoint"""

    def __init__(self):
        self.storage = get_storage()

    def on_get_lpi(self, req: Request, resp: Response, experiment_name: str):
        """
        Handle GET requests for plotting lpi plots on plots/lpi/:experiment
        where ``experiment`` is the user-defined name of the experiment.
        """
        experiment = ExperimentClient(retrieve_experiment(experiment_name), None)
        resp.body = experiment.plot.lpi().to_json()

    def on_get_parallel_coordinates(
        self, req: Request, resp: Response, experiment_name: str
    ):
        """
        Handle GET requests for plotting parallel coordinates plots on
        plots/parallel_coordinates/:experiment where ``experiment`` is the user-defined name of the
        experiment.
        """
        experiment = ExperimentClient(retrieve_experiment(experiment_name), None)
        resp.body = experiment.plot.parallel_coordinates().to_json()

    def on_get_partial_dependencies(
        self, req: Request, resp: Response, experiment_name: str
    ):
        """
        Handle GET requests for plotting partial dependencies on
        plots/partial_dependencies/:experiment where ``experiment`` is the user-defined name of the
        experiment.
        """
        experiment = ExperimentClient(retrieve_experiment(experiment_name), None)
        resp.body = experiment.plot.partial_dependencies().to_json()

    def on_get_regret(self, req: Request, resp: Response, experiment_name: str):
        """
        Handle GET requests for plotting regret plots on plots/regret/:experiment
        where ``experiment`` is the user-defined name of the experiment.
        """
        experiment = ExperimentClient(retrieve_experiment(experiment_name), None)
        resp.body = experiment.plot.regret().to_json()
