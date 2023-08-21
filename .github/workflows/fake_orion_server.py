#!/usr/bin/env python
# pylint: disable=too-few-public-methods
"""
Fake web application endpoint
=============================

Starts a http endpoint to serve pre-recorded responses to requests

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
    """Resource class to serve frontend files."""

    JSON_PATH = os.path.join(os.path.dirname(__file__), "fake_orion_server.json")

    def __init__(self):
        with open(self.JSON_PATH) as file:
            self.results = json.load(file)
        logger.info("Loaded fake server.")

    def on_get(self, req, resp, *args, **kwargs):
        """Hack HTML and Javascript files to set up backend if necessary."""
        # path = req.uri.replace("127.0.0.1:8001", "127.0.0.1:8000")
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


class Aaaa(falcon.App):
    def __init__(self):
        frontends_uri = ["http://localhost:3000", "http://127.0.0.1:3000"]
        cors = MyCORS(allow_origins_list=frontends_uri)
        super().__init__(middleware=[cors.middleware])

        resource = StaticResource()
        ##
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
        ##
        self.add_sink(resource.on_get)


def main():
    """Starts a fake server to serve pre-recorded responses to http requests"""
    app = Aaaa()
    gunicorn_app = GunicornApp(app)
    gunicorn_app.run()


if __name__ == "__main__":
    main()
