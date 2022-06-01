#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web application endpoint
========================

Starts an http endpoint to serve requests

"""
import logging

from gunicorn.app.base import BaseApplication

log = logging.getLogger(__name__)
DESCRIPTION = "Starts Oríon Dashboard"


import os
import falcon
import mimetypes


class StaticResource:
    STATIC_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'dashboard', 'build'))

    PLACEHOLDER = "window.__ORION_BACKEND__"
    TEXT_TYPES = ("text/html", "application/javascript")

    def __init__(self, args):
        self.backend = args.get("backend", None)

    def on_get(self, req, resp):
        path = req.relative_uri.strip("/") or "index.html"
        file_path = os.path.join(self.STATIC_DIR, path)
        if os.path.isfile(file_path):
            content_type, _ = mimetypes.guess_type(file_path)
            resp.status = falcon.HTTP_200
            resp.content_type = content_type
            with open(file_path, "rb") as file:
                content = file.read()
                if content_type in self.TEXT_TYPES:
                    content = content.decode()
                    if self.backend is not None and self.PLACEHOLDER in content:
                        content = content.replace(self.PLACEHOLDER, repr(self.backend))
                resp.body = content
        else:
            resp.status = falcon.HTTP_404


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    serve_parser = parser.add_parser("frontend", help=DESCRIPTION, description=DESCRIPTION)

    serve_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="port to run frontend (default 8000)"
    )

    serve_parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default='http://127.0.0.1:8000',
        help="backend address (default: http://127.0.0.1:8000)",
    )

    serve_parser.set_defaults(func=main)

    return serve_parser


def main(args):
    """Starts an application server to serve http requests"""
    app = falcon.API()
    resource = StaticResource(args)
    app.add_sink(resource.on_get)

    gunicorn_app = GunicornApp(app, args)
    gunicorn_app.run()


class GunicornApp(BaseApplication):
    """Custom Gunicorn application, required when integrating gunicorn as an API."""

    def __init__(self, app, args=None):
        options = {}
        if args:
            options['bind'] = f"localhost:{args['port']}"
        self.options = options
        self.application = app
        super(GunicornApp, self).__init__()

    def init(self, parser, opts, args):
        """Pre-run initialization"""
        pass

    def load_config(self):
        """Load the gunicorn config"""
        for key, value in self.options.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        """Load the WSGI application"""
        return self.application
