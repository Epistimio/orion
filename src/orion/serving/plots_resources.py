#!/usr/bin/env python
"""
Module responsible for the plots/ REST endpoint
================================================

Serves all the requests made to plots/ REST endpoint.

"""
import base64
import json

import numpy as np
from falcon import Request, Response

from orion.client import ExperimentClient
from orion.serving.parameters import retrieve_experiment

# Mapping from plotly.js typed-array short dtype codes to numpy dtype strings.
_DTYPE_MAP = {
    "f8": "float64",
    "f4": "float32",
    "i4": "int32",
    "i2": "int16",
    "i1": "int8",
    "u4": "uint32",
    "u2": "uint16",
    "u1": "uint8",
}


def _decode_b64(obj):
    """Recursively decode plotly b64-encoded typed arrays into plain lists.

    Plotly >= 6 encodes numpy arrays as ``{"dtype": ..., "bdata": ...}``
    which plotly.js < 3 cannot read.
    """
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:
            dtype = _DTYPE_MAP.get(obj["dtype"], obj["dtype"])
            return np.frombuffer(base64.b64decode(obj["bdata"]), dtype=dtype).tolist()
        return {k: _decode_b64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_b64(item) for item in obj]
    return obj


def _figure_to_json(fig):
    """Serialize a plotly Figure to JSON compatible with plotly.js 2.x."""
    return json.dumps(_decode_b64(json.loads(fig.to_json())))


class PlotsResource:
    """Serves all the requests made to plots/ REST endpoint"""

    def __init__(self, storage):
        self.storage = storage

    def on_get_lpi(self, req: Request, resp: Response, experiment_name: str):
        """
        Handle GET requests for plotting lpi plots on plots/lpi/:experiment
        where ``experiment`` is the user-defined name of the experiment.
        """
        experiment = ExperimentClient(
            retrieve_experiment(self.storage, experiment_name), None
        )
        resp.text = _figure_to_json(experiment.plot.lpi())

    def on_get_parallel_coordinates(
        self, req: Request, resp: Response, experiment_name: str
    ):
        """
        Handle GET requests for plotting parallel coordinates plots on
        plots/parallel_coordinates/:experiment where ``experiment`` is the user-defined name of the
        experiment.
        """
        experiment = ExperimentClient(
            retrieve_experiment(self.storage, experiment_name), None
        )
        resp.text = _figure_to_json(experiment.plot.parallel_coordinates())

    def on_get_partial_dependencies(
        self, req: Request, resp: Response, experiment_name: str
    ):
        """
        Handle GET requests for plotting partial dependencies on
        plots/partial_dependencies/:experiment where ``experiment`` is the user-defined name of the
        experiment.
        """
        experiment = ExperimentClient(
            retrieve_experiment(self.storage, experiment_name), None
        )
        resp.text = _figure_to_json(experiment.plot.partial_dependencies())

    def on_get_regret(self, req: Request, resp: Response, experiment_name: str):
        """
        Handle GET requests for plotting regret plots on plots/regret/:experiment
        where ``experiment`` is the user-defined name of the experiment.
        """
        experiment = ExperimentClient(
            retrieve_experiment(self.storage, experiment_name), None
        )
        resp.text = _figure_to_json(experiment.plot.regret())
