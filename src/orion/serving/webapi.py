#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.serving.webapi` -- WSGI REST server application
===========================================================

.. module:: webapi
   :platform: Unix
   :synopsis: WSGI REST server application
"""

import falcon

from orion.storage.base import setup_storage
from orion.serving.experiments_resource import ExperimentsResource
from orion.serving.runtime import RuntimeResource


class WebApi(falcon.API):
    """Main entry point into a Falcon-based app.
    An instance provides a callable WSGI interface and a routing
    engine.
    """

    def __init__(self, config=None):
        super(WebApi, self).__init__()
        self.config = config

        setup_storage(config.get('storage'))

        # Create our resources
        experiments_endpoint = ExperimentsResource()
        root_endpoint = RuntimeResource()

        # Build routes
        self.add_route('/', root_endpoint)
        self.add_route('/experiments', experiments_endpoint)

    def start(self):
        """A hook to when a Gunicorn worker calls run()."""
        pass

    def stop(self, signal):
        """A hook to when a Gunicorn worker starts shutting down."""
        pass
