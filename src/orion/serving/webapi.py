#!/usr/bin/env python
"""
WSGI REST server application
============================

Exposes a WSGI REST server application instance by subclassing ``falcon.API``.

"""

import logging

import falcon

from orion.serving.benchmarks_resource import BenchmarksResource
from orion.serving.experiments_resource import ExperimentsResource
from orion.serving.plots_resources import PlotsResource
from orion.serving.runtime import RuntimeResource
from orion.serving.storage_resource import StorageResource
from orion.serving.trials_resource import TrialsResource

logger = logging.getLogger(__name__)


class OriginEnforcerMiddleware:
    """Reject requests from disallowed origins with 403 Forbidden.

    Falcon's built-in CORSMiddleware handles CORS headers but does not
    reject requests from disallowed origins — it simply omits the headers.
    This middleware enforces strict origin checking: if a request includes
    an Origin header that is not in the allowed list, it is rejected.

    Requests without an Origin header (e.g. same-origin browser requests
    or non-browser clients) are allowed through.
    """

    def __init__(self, allow_origins):
        self.allow_origins = set(allow_origins)

    def process_resource(self, req, resp, resource, params):
        """Generate a 403 Forbidden response if origin is not allowed."""
        origin = req.get_header("origin")
        if origin and origin not in self.allow_origins:
            raise falcon.HTTPForbidden()


class WebApi(falcon.App):
    """
    Main entry point into a Falcon-based app. An instance provides a callable WSGI interface and a
    routing engine.
    """

    def __init__(self, storage, config=None):
        # By default, server will reject requests coming from a server
        # with different origin. E.g., if server is hosted at
        # http://myorionserver.com, it won't accept an API call
        # coming from a server not hosted at same address
        # (e.g. a local installation at http://localhost)
        # Cross-Origin Resource Sharing (CORS) security:
        # https://developer.mozilla.org/fr/docs/Web/HTTP/CORS
        frontends_uri = (
            config["frontends_uri"]
            if "frontends_uri" in config
            else ["http://localhost:3000"]
        )
        logger.info(
            "allowed frontends: {}".format(
                ", ".join(frontends_uri) if frontends_uri else "(none)"
            )
        )
        cors = falcon.CORSMiddleware(allow_origins=frontends_uri)
        origin_enforcer = OriginEnforcerMiddleware(frontends_uri)
        super().__init__(middleware=[origin_enforcer, cors])
        self.config = config
        self.storage = storage

        # Create our resources
        root_resource = RuntimeResource(self.storage)
        experiments_resource = ExperimentsResource(self.storage)
        benchmarks_resource = BenchmarksResource(self.storage)
        trials_resource = TrialsResource(self.storage)
        plots_resource = PlotsResource(self.storage)
        storage_resource = StorageResource(self.storage)

        # Build routes
        self.add_route("/", root_resource)
        self.add_route("/experiments", experiments_resource)
        self.add_route("/experiments/{name}", experiments_resource, suffix="experiment")
        self.add_route("/benchmarks", benchmarks_resource)
        self.add_route("/benchmarks/{name}", benchmarks_resource, suffix="benchmark")
        self.add_route(
            "/experiments/status/{name}",
            experiments_resource,
            suffix="experiment_status",
        )
        self.add_route(
            "/trials/{experiment_name}", trials_resource, suffix="trials_in_experiment"
        )
        self.add_route(
            "/trials/{experiment_name}/{trial_id}",
            trials_resource,
            suffix="trial_in_experiment",
        )
        self.add_route(
            "/trials/{experiment_name}/{trial_id}/set-status/{status}",
            trials_resource,
            suffix="trial_set_status_in_experiment",
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
        self.add_route("/dump", storage_resource, suffix="dump")
        self.add_route("/load", storage_resource, suffix="load")
        self.add_route(
            "/import-status/{name}", storage_resource, suffix="import_status"
        )

    def start(self):
        """A hook to when a Gunicorn worker calls run()."""

    def stop(self, signal):
        """A hook to when a Gunicorn worker starts shutting down."""
