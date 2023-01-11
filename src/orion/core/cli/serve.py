#!/usr/bin/env python
"""
Web application endpoint
========================

Starts an http endpoint to serve requests

"""
import argparse
import logging

from gunicorn.app.base import BaseApplication

from orion.core.io import experiment_builder
from orion.serving.webapi import WebApi
from orion.storage.base import setup_storage

log = logging.getLogger(__name__)
DESCRIPTION = "Starts Oríon's REST API server"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    serve_parser = parser.add_parser("serve", help=DESCRIPTION, description=DESCRIPTION)

    serve_parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="user provided " "orion configuration file",
    )

    serve_parser.set_defaults(func=main)

    return serve_parser


def main(args):
    """Starts an application server to serve http requests"""
    config = experiment_builder.get_cmd_config(args)
    gunicorn_config = config.get("gunicorn", {})

    storage = setup_storage(config.get("storage"))
    web_api = WebApi(storage, config)

    gunicorn_app = GunicornApp(web_api, options=gunicorn_config)
    gunicorn_app.run()


class GunicornApp(BaseApplication):
    """Custom Gunicorn application, required when integrating gunicorn as an API."""

    def __init__(self, app, options=None):
        self.options = options or {}
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
