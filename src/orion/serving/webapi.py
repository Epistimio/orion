# -*- coding: utf-8 -*-
"""
:mod:`orion.serving` -- Entry point for web api service
==============================================================

.. module:: webapi
   :platform: Unix
   :synopsis: Entry point for web api service

"""
import falcon

import orion.core.io.experiment_builder as experiment_builder
from orion.serving.experiments_resource import ExperimentsResource


experiment_builder.setup_storage({
    "type": "legacy",
    'database': {'type': 'PickledDB', "host": "../database.pkl"}
})


def create():
    """Create a WSGI compatible app object"""
    experiments_endpoint = ExperimentsResource()

    app = falcon.API()
    app.add_route('/experiments', experiments_endpoint)
    return app


app = create()
