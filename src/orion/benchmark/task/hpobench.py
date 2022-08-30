"""
Task for HPOBench
=================
"""
import importlib
import subprocess
from typing import Dict, List, Union

from orion.algo.space.configspace import to_orionspace
from orion.benchmark.task.base import BenchmarkTask


class HPOBench(BenchmarkTask):
    """Benchmark Task wrapper over HPOBench (https://github.com/automl/HPOBench)

    For more information on HPOBench, see original paper at https://arxiv.org/abs/2109.06716.

    Katharina Eggensperger, Philipp Müller, Neeratyoy Mallik, Matthias Feurer, René Sass, Aaron Klein,
    Noor Awad, Marius Lindauer, Frank Hutter. "HPOBench: A Collection of Reproducible Multi-Fidelity
    Benchmark Problems for HPO" Thirty-fifth Conference on Neural Information Processing Systems
    Datasets and Benchmarks Track (Round 2).

    Parameters
    ----------
    max_trials : int
        Maximum number of trials for this task.
    hpo_benchmark_class : str
        Full path to a particular class of benchmark in HPOBench.
    benchmark_kwargs: str
        Optional parameters to create benchmark instance of class `hpo_benchmark_class`.
    objective_function_kwargs: dict
        Optional parameters to use when calling `objective_function` of the benchmark instance.
    """

    def __init__(
        self,
        max_trials: int,
        hpo_benchmark_class: Union[str, None] = None,
        benchmark_kwargs: dict = dict(),
        objective_function_kwargs: dict = dict(),
    ):
        super().__init__(
            max_trials=max_trials,
            hpo_benchmark_class=hpo_benchmark_class,
            benchmark_kwargs=benchmark_kwargs,
            objective_function_kwargs=objective_function_kwargs,
        )
        self._verify_benchmark(hpo_benchmark_class)
        self.hpo_benchmark_cls = self._load_benchmark(hpo_benchmark_class)
        self.benchmark_kwargs = benchmark_kwargs
        self.objective_function_kwargs = objective_function_kwargs

    def call(self, **kwargs) -> List[Dict]:
        hpo_benchmark = self.hpo_benchmark_cls(**self.benchmark_kwargs)
        result_dict = hpo_benchmark.objective_function(
            configuration=kwargs, **self.objective_function_kwargs
        )
        objective = result_dict["function_value"]
        return [
            dict(
                name=self.hpo_benchmark_cls.__name__, type="objective", value=objective
            )
        ]

    def _load_benchmark(self, hpo_benchmark_class: str):
        package, cls = hpo_benchmark_class.rsplit(".", 1)
        module = importlib.import_module(package)
        return getattr(module, cls)

    def _verify_benchmark(self, hpo_benchmark_class: str):
        if not hpo_benchmark_class:
            raise AttributeError("Please provide full path to a HPOBench benchmark")
        if "container" in hpo_benchmark_class:
            code, message = subprocess.getstatusoutput("singularity -h")
            if code != 0:
                raise AttributeError(
                    "Can not run conterized benchmark without Singularity: {}".format(
                        message
                    )
                )

    def get_search_space(self) -> Dict[str, str]:
        configuration_space = self.hpo_benchmark_cls(
            **self.benchmark_kwargs
        ).get_configuration_space()
        return to_orionspace(configuration_space)
