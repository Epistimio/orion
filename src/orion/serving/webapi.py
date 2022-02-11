#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WSGI REST server application
============================

Exposes a WSGI REST server application instance by subclassing ``falcon.API``.

"""

import falcon

from orion.serving.experiments_resource import ExperimentsResource
from orion.serving.plots_resources import PlotsResource
from orion.serving.runtime import RuntimeResource
from orion.serving.trials_resource import TrialsResource
from orion.storage.base import setup_storage


class WebApi(falcon.API):
    """
    Main entry point into a Falcon-based app. An instance provides a callable WSGI interface and a
    routing engine.
    """

    def __init__(self, config=None):
        super(WebApi, self).__init__()
        self.config = config

        setup_storage(config.get("storage"))

        # Create our resources
        root_resource = RuntimeResource()
        experiments_resource = ExperimentsResource()
        trials_resource = TrialsResource()
        plots_resource = PlotsResource()

        # Build routes
        self.add_route("/", root_resource)
        self.add_route("/experiments", experiments_resource)
        self.add_route("/experiments/{name}", experiments_resource, suffix="experiment")
        self.add_route(
            "/trials/{experiment_name}", trials_resource, suffix="trials_in_experiment"
        )
        self.add_route(
            "/trials/{experiment_name}/{trial_id}",
            trials_resource,
            suffix="trial_in_experiment",
        )
        self.add_route("/plots/lpi/{experiment_name}", plots_resource, suffix="lpi")
        self.add_route(
            "/plots/partial_dependencies/{experiment_name}",
            plots_resource,
            suffix="partial_dependencies",
        )
        self.add_route(
            "/plots/parallel_coordinates/{experiment_name}",
            plots_resource,
            suffix="parallel_coordinates",
        )
        self.add_route(
            "/plots/regret/{experiment_name}", plots_resource, suffix="regret"
        )

    def start(self):
        """A hook to when a Gunicorn worker calls run()."""
        pass

    def stop(self, signal):
        """A hook to when a Gunicorn worker starts shutting down."""
        pass
