"""
Module responsible for the benchmarks/ REST endpoint
=====================================================

Serves all the requests made to benchmarks/ REST endpoint.

"""
import json

from falcon import Request, Response

from orion.serving.parameters import retrieve_benchmark, verify_query_parameters
from orion.serving.responses import build_benchmark_response, build_benchmarks_response


class BenchmarksResource:
    """Handle requests for the benchmarks/ REST endpoint"""

    def __init__(self, storage):
        self.storage = storage

    def on_get(self, req: Request, resp: Response):
        """Handle the GET requests for benchmarks/"""
        benchmarks = self.storage.fetch_benchmark({})

        response = build_benchmarks_response(benchmarks)
        resp.text = json.dumps(response)

    def on_get_benchmark(self, req: Request, resp: Response, name: str):
        """
        Handle GET requests for benchmarks/:name where `name` is
        the user-defined name of the benchmark
        """
        verify_query_parameters(req.params, ["assessment", "task", "algorithms"])
        assessment = req.get_param("assessment")
        task = req.get_param("task")
        algorithms = req.get_param_as_list("algorithms")
        benchmark = retrieve_benchmark(
            self.storage,
            name,
            assessment=assessment,
            task=task,
            algorithms=algorithms,
        )

        response = build_benchmark_response(benchmark, assessment, task, algorithms)
        resp.text = json.dumps(response)
