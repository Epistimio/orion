# -*- coding: utf-8 -*-
"""
Represent the '/' REST endpoint
===============================
"""
import json

import orion.core
from orion.storage.base import get_storage


class RuntimeResource(object):
    """Handle requests for the '/' REST endpoint"""

    def __init__(self):
        pass

    def on_get(self, req, resp):
        """Handle the HTTP GET requests for the '/' endpoint

        Parameters
        ----------
        req
            The request
        resp
            The response to send back
        """
        database = get_storage()._db.__class__.__name__
        response = {
            "orion": orion.core.__version__,
            "server": "gunicorn",
            "database": database,
        }

        resp.body = json.dumps(response, indent=4)
