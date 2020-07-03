# -*- coding: utf-8 -*-
"""
:mod:`orion.serving.experiments_resource` -- Represent the experiments/ REST endpoint
=====================================================================================

.. module:: experiments_resource
   :platform: Unix
   :synopsis: Represent the experiments/ REST endpoint

"""
import json

from orion.storage.base import get_storage


class ExperimentsResource(object):
    """Handle requests for the experiments/ REST endpoint"""

    def __init__(self):
        self.storage = get_storage()

    def on_get(self, req, resp):
        """Handle the HTTP GET requests

        Parameters
        ----------
        req
            The request
        resp
            The response to send back
        """
        experiments = self.storage.fetch_experiments({})
        leaf_experiments = self._find_latest_versions(experiments)

        result = []
        for name, version in leaf_experiments.items():
            result.append({
                'name': name,
                'version': version
            })

        resp.body = json.dumps(result, indent=4)

    def _find_latest_versions(self, experiments):
        """Find the latest versions of the experiments"""
        leaf_experiments = {}
        for experiment in experiments:
            name = experiment['name']
            version = experiment['version']

            if name in leaf_experiments:
                leaf_experiments[name] = max(leaf_experiments[name], version)
            else:
                leaf_experiments[name] = version
        return leaf_experiments
