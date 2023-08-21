#!/usr/bin/env python
# pylint: disable=too-few-public-methods
"""
Fake web application endpoint
=============================

Starts a http endpoint to serve pre-recorded responses to Orion Web API

Dashboard SRC tests always succeed on a local machine,
but fail sometimes on Github Actions because of request timeouts.
It seems CI does not provide enough resource to allow Orion server to
always respond to requests in a reasonable time.

It's currently difficult to find exactly which part of code leads to request timeouts.
However, Orion web API already has dedicated Python tests, and the goal of dashboard tests
is mainly to test dashboard, ie. interface, UI and how it reacts to request responses.
So, testing the dashboard should not need to have a real running Orion server:
we just need to mock the server with coherent fake responses the dashboard could use.

The purpose of this script is to launch a very simple fake orion server,
which just answers to requests with pre-recorded responses. It should be
light and fast enough to run on CI with the least resource consumption,
thus allowing dashboard tests to always pass.

Pre-recorded responses are stored in a dictionary in JSON file `fake_orion_server.json`.
Each response is stored as a key-value in the dictionary, where:
- key is the address (API call) that must lead to the response
- value is a dictionary with format {"t": float, "r": data}, where:
  - "t" is the response time (in seconds) that was needed to get response on a local machine.
    Response time can be used by fake server to simulate delay in server response.
  - "r" is the actual response, retrieved on a local machine.

Current JSON file contains only the responses expected for current tests.
It should be updated for any new API calls added to dashboard tests.
"""
import json
import logging
import os
import time

import falcon
from gunicorn.app.base import BaseApplication

from orion.serving.webapi import MyCORS

logger = logging.getLogger(__name__)


class StaticResource:
    """Resource class to serve responses."""

    JSON_PATH = os.path.join(os.path.dirname(__file__), "fake_orion_server.json")

    def __init__(self):
        with open(self.JSON_PATH) as file:
            self.results = json.load(file)
        logger.info("Loaded fake server.")

    def on_get(self, req, resp, *args, **kwargs):
        """Return recorded response for given resource."""
        path = req.uri
        if path in self.results:
            result = self.results[path]
            duration = result["t"]
            data = result["r"]
            time.sleep(duration)
            resp.text = json.dumps(data)
        else:
            resp.status = falcon.HTTP_404


class GunicornApp(BaseApplication):
    """Custom Gunicorn application, required when integrating gunicorn as an API."""

    def __init__(self, app):
        # Hardcoded options.
        self.options = {
            "bind": "127.0.0.1:8000",
            "workers": 1,
            "threads": 1,
            "timeout": 600,
            "loglevel": "debug",
            "worker_tmp_dir": "gunicorn_tmp_dir",
        }
        self.application = app
        super().__init__()

    def init(self, parser, opts, args):
        """Pre-run initialization"""

    def load_config(self):
        """Load the gunicorn config"""
        for key, value in self.options.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        """Load the WSGI application"""
        return self.application


class FalconApp(falcon.App):
    """Falcon application to simulate Orion Web API endpoints."""

    def __init__(self):
        """Initialize."""

        # Prevent CORS (Cross-Origin) issues.
        frontends_uri = ["http://localhost:3000", "http://127.0.0.1:3000"]
        cors = MyCORS(allow_origins_list=frontends_uri)
        super().__init__(middleware=[cors.middleware])

        # Create static resource and map it to endpoints.
        resource = StaticResource()
        # NB: Mapping could have been done with a single line `self.add_sink()`,
        # but it seems to ignore CORS workaround above,
        # making frontend requests rejected by server.
        # So, we just map resource to each endpoint individually.
        self.add_route("/", resource)
        self.add_route("/experiments", resource)
        self.add_route("/experiments/{name}", resource)
        self.add_route("/benchmarks", resource)
        self.add_route("/benchmarks/{name}", resource)
        self.add_route(
            "/experiments/status/{name}",
            resource,
        )
        self.add_route("/trials/{experiment_name}", resource)
        self.add_route(
            "/trials/{experiment_name}/{trial_id}",
            resource,
        )
        self.add_route(
            "/trials/{experiment_name}/{trial_id}/set-status/{status}",
            resource,
        )
        self.add_route("/plots/lpi/{experiment_name}", resource)
        self.add_route("/plots/partial_dependencies/{experiment_name}", resource)
        self.add_route(
            "/plots/parallel_coordinates/{experiment_name}",
            resource,
        )
        self.add_route("/plots/regret/{experiment_name}", resource)
        self.add_route("/dump", resource)
        self.add_route("/load", resource)
        self.add_route("/import-status/{name}", resource)
        self.add_sink(resource.on_get)


def main():
    """Starts a fake server to serve pre-recorded responses to http requests"""
    app = FalconApp()
    gunicorn_app = GunicornApp(app)
    gunicorn_app.run()


if __name__ == "__main__":
    main()
