"""
Represent the '/' REST endpoint
===============================
"""
import json

import orion.core


class RuntimeResource:
    """Handle requests for the '/' REST endpoint"""

    def __init__(self, storage):
        self.storage = storage

    def on_get(self, req, resp):
        """Handle the HTTP GET requests for the '/' endpoint

        Parameters
        ----------
        req
            The request
        resp
            The response to send back
        """
        database = self.storage._db.__class__.__name__
        response = {
            "orion": orion.core.__version__,
            "server": "gunicorn",
            "database": database,
        }

        resp.text = json.dumps(response, indent=4)
