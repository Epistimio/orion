#!/usr/bin/env python
"""
WSGI REST server application
============================

Exposes a WSGI REST server application instance by subclassing ``falcon.API``.

"""

import logging

import falcon
from falcon_cors import CORS, CORSMiddleware

from orion.serving.benchmarks_resource import BenchmarksResource
from orion.serving.experiments_resource import ExperimentsResource
from orion.serving.plots_resources import PlotsResource
from orion.serving.runtime import RuntimeResource
from orion.serving.trials_resource import TrialsResource

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MyCORSMiddleware(CORSMiddleware):
    """Subclass of falcon-cors CORSMiddleware class.

    Generate a HTTP 403 Forbidden response if request sender is not allowed
    to access requested content.

    Default middleware just prints a message in server side
    (e.g. "Aborting response due to origin not allowed"), but still
    sends content, so, a client ignoring headers might still access
    data even if not allowed.

    CORS middleware role is to add necessary "access-control-" headers to
    response to mark it as allowed. So, a response lacking expected headers
    after call to parent method `process_ressource()` can be considered
    to not be delivered to request sender.

    More info about CORS:
    - https://developer.mozilla.org/fr/docs/Web/HTTP/CORS
    - https://fr.wikipedia.org/wiki/Cross-origin_resource_sharing
    """

    def process_resource(self, req, resp, resource, *args):
        """Generate a 403 Forbidden response if response is not allowed."""

        cors_resp_headers_before = [
            header
            for header in resp.headers
            if header.lower().startswith("access-control-")
        ]
        assert not cors_resp_headers_before, cors_resp_headers_before

        super().process_resource(req, resp, resource, *args)

        # We then verify if some access control headers were added to response.
        # If not, response is not allowed.
        # Special case: if request did not have an origin, it was certainly sent from
        # a browser (ie. not another server), so CORS is not relevant.
        cors_resp_headers_after = [
            header
            for header in resp.headers
            if header.lower().startswith("access-control-")
        ]
        if not cors_resp_headers_after and req.get_header("origin"):
            raise falcon.HTTPForbidden()


class MyCORS(CORS):
    """Subclass of falcon-cors CORS class to return a custom middleware."""

    @property
    def middleware(self):
        return MyCORSMiddleware(self)


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
        # This is a Cross-Origin Resource Sharing (CORS) security:
        # https://developer.mozilla.org/fr/docs/Web/HTTP/CORS
        # To make server accept CORS requests, we need to use
        # falcon-cors package: https://github.com/lwcolton/falcon-cors
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
        cors = MyCORS(allow_origins_list=frontends_uri)
        super().__init__(middleware=[cors.middleware])
        self.config = config
        self.storage = storage

        # Create our resources
        root_resource = RuntimeResource(self.storage)
        experiments_resource = ExperimentsResource(self.storage)
        benchmarks_resource = BenchmarksResource(self.storage)
        trials_resource = TrialsResource(self.storage)
        plots_resource = PlotsResource(self.storage)

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

    def stop(self, signal):
        """A hook to when a Gunicorn worker starts shutting down."""
