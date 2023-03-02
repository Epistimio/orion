#!/usr/bin/env python
# pylint: disable=too-few-public-methods
"""
Web application endpoint
========================

Starts an http endpoint to serve requests

"""
import logging
import mimetypes
import os
import site
import sys

import falcon
from gunicorn.app.base import BaseApplication

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DESCRIPTION = "Starts Oríon Dashboard"


def get_dashboard_build_path():
    """Find dashboard build folder.

    If package is installed, dashboard build should be in installation prefix
    https://docs.python.org/3/distutils/setupscript.html#installing-additional-files
    Otherwise, dashboard build should be in dashboard folder near src
    in orion repository.

    NB (2023/02/20): It seems that, depending on installation, additional files may
    be installed in `<sys.prefix>/local` instead of just `<sys.prefix>`.
    More info:
    https://stackoverflow.com/questions/14211575/any-python-function-to-get-data-files-root-directory#comment99087548_14211600
    """
    current_file_path = __file__
    if current_file_path.startswith(sys.prefix):
        dashboard_build_path = os.path.join(sys.prefix, "orion-dashboard", "build")
        if not os.path.isdir(dashboard_build_path):
            dashboard_build_path = os.path.join(
                sys.prefix, "local", "orion-dashboard", "build"
            )
    elif current_file_path.startswith(site.USER_BASE):
        dashboard_build_path = os.path.join(site.USER_BASE, "orion-dashboard", "build")
        if not os.path.isdir(dashboard_build_path):
            dashboard_build_path = os.path.join(
                site.USER_BASE, "local", "orion-dashboard", "build"
            )
    else:
        dashboard_build_path = os.path.abspath(
            os.path.join(
                os.path.dirname(current_file_path),
                "..",
                "..",
                "..",
                "..",
                "dashboard",
                "build",
            )
        )
    if not os.path.isdir(dashboard_build_path):
        raise RuntimeError(
            f"Cannot find dashboard static files to run frontend. "
            f"Expected to be located at: {dashboard_build_path}"
        )
    return dashboard_build_path


class StaticResource:
    """Resource class to serve frontend files."""

    STATIC_DIR = get_dashboard_build_path()
    PLACEHOLDER = "window.__ORION_BACKEND__"
    TEXT_TYPES = ("text/html", "application/javascript")

    def __init__(self, args):
        self.backend = args.get("backend", None)
        logger.info(f"Dashboard build located at: {self.STATIC_DIR}")

    def on_get(self, req, resp):
        """Hack HTML and Javascript files to setup backend if necessary."""
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
    serve_parser = parser.add_parser(
        "frontend", help=DESCRIPTION, description=DESCRIPTION
    )

    serve_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=3000,
        help="port to run frontend (default 3000)",
    )

    serve_parser.add_argument(
        "-b",
        "--backend",
        type=str,
        default="http://127.0.0.1:8000",
        help="backend address (default: http://127.0.0.1:8000)",
    )

    serve_parser.set_defaults(func=main)

    return serve_parser


def main(args):
    """Starts an application server to serve http requests"""
    app = falcon.App()
    resource = StaticResource(args)
    app.add_sink(resource.on_get)

    gunicorn_app = GunicornApp(app, args)
    gunicorn_app.run()


class GunicornApp(BaseApplication):
    """Custom Gunicorn application, required when integrating gunicorn as an API."""

    def __init__(self, app, args=None):
        options = {}
        if args:
            options["bind"] = f"localhost:{args['port']}"
        self.options = options
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
