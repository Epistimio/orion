"""
Helpers for building responses according to the specification
=============================================================

Offers functions and attributes to generate response objects according to the API specification.

"""
from typing import List

from orion.benchmark import Benchmark
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial

ERROR_EXPERIMENT_NOT_FOUND = "Experiment not found"
ERROR_INVALID_PARAMETER = "Invalid parameter"
ERROR_TRIAL_NOT_FOUND = "Trial not found"
ERROR_BENCHMARK_NOT_FOUND = "Benchmark not found"
ERROR_BENCHMARK_STUDY_NOT_FOUND = "Benchmark study not found"


def build_trial_response(trial: Trial) -> dict:
    """
    Build the response representing a trial according to the API specification.

    Parameters
    ----------
    trial: Trial
        The trial to return for the API.

    Returns
    -------
    A JSON-serializable dict representing the given trial.

    """
    return {
        "id": trial.id,
        "submitTime": str(trial.submit_time),
        "startTime": str(trial.start_time),
        "endTime": str(trial.end_time),
        "parameters": trial.params,
        "objective": trial.objective.value if trial.status == "completed" else None,
        "statistics": {
            statistic.name: statistic.value for statistic in trial.statistics
        },
        "status": trial.status,
    }


def build_experiment_response(
    experiment: Experiment, status: str, algorithm: dict, best_trial: Trial = None
):
    """
    Build the response representing an experiment response object according to the API
    specification.

    Parameters
    ----------
    experiment: Experiment
        The experiment to return to the API
    status: str
        The status of the experiment
    algorithm: dict
        The dictionary containing the algorithm's configuration
    best_trial: Trial (Optional)
        The best trial to date of the experiment

    Returns
    -------
    A JSON-serializable experiment response object representing the given experiment.
    """

    data = {
        "name": experiment.name,
        "version": experiment.version,
        "status": status,
        "trialsCompleted": 0,
        "startTime": None,
        "endTime": None,
        "user": experiment.metadata["user"],
        "orionVersion": experiment.metadata["orion_version"],
        "config": {
            "maxTrials": experiment.max_trials,
            "maxBroken": experiment.max_broken,
            "algorithm": algorithm,
            "space": experiment.configuration["space"],
        },
        "bestTrial": build_trial_response(best_trial) if best_trial else {},
    }

    stats = experiment.stats
    if stats.trials_completed:
        data["trialsCompleted"] = stats.trials_completed
        data["startTime"] = str(stats.start_time)
        data["endTime"] = str(stats.finish_time)

    return data


def build_experiments_response(experiments: dict):
    """
    Build the response representing a list of experiments according to the API specification.

    Parameters
    ----------
    experiments: dict
        A dict containing pairs of ``experiment-name:experiment-version``.

    Returns
    -------
    A JSON-serializable list of experiments as defined in the API specification.
    """
    result = []
    for name, version in experiments.items():
        result.append({"name": name, "version": version})
    return result


def build_trials_response(trials: list):
    """
    Build the response representing a list of trials according to the API specification.

    Parameters
    ----------
    trials: list
        A list of :class:`orion.core.worker.trial.Trial`.

    Returns
    -------
    A JSON-serializable list of trials as defined in the API specification.
    """
    response = []
    for trial in trials:
        response.append({"id": trial.id})
    return response


def build_benchmark_response(
    benchmark: Benchmark,
    assessment: str = None,
    task: str = None,
    algorithms: List[str] = None,
):
    """
    Build the response representing an experiment response object according to the API
    specification.

    Parameters
    ----------
    benchmark: Benchmark
        The benchmark to return to the API
    assessment: str
        The assessment analysis to return
    task: str
        The task analysis to return
    algorithms: list
        The list of algorithm names to include in the analysis.

    Returns
    -------
    A JSON-serializable benchmark response object representing the given benchmark.
    """

    def convert_plotly_to_json(analysis):
        for study_name, study_analysis in analysis.items():
            for task_name, task_analysis in study_analysis.items():
                analysis[study_name][task_name] = {
                    key: figure.to_json() for key, figure in task_analysis.items()
                }
        return analysis

    data = {
        "name": benchmark.name,
        "algorithms": benchmark.algorithms,
        "tasks": [task.configuration for task in benchmark.targets[0]["task"]],
        "assessments": [
            assessment.configuration for assessment in benchmark.targets[0]["assess"]
        ],
        "analysis": convert_plotly_to_json(
            benchmark.analysis(assessment=assessment, task=task, algorithms=algorithms)
        ),
    }

    return data


def build_benchmarks_response(benchmarks: dict):
    """
    Build the response representing a list of benchmarks according to the API specification.

    Parameters
    ----------
    benchmarks: dict
        A dict containing pairs of ``benchmark-name: {algorithms:[], tasks:[], assessments:[]}``.

    Returns
    -------
    A JSON-serializable list of benchmarks as defined in the API specification.
    """
    result = []
    for benchmark in benchmarks:
        benchmark_response = dict(
            name=benchmark["name"],
            algorithms=benchmark["algorithms"],
            tasks=benchmark["targets"][0]["task"],
        )

        assessments = {}
        for target in benchmark["targets"]:
            assessments.update(target["assess"])

        benchmark_response["assessments"] = assessments

        result.append(benchmark_response)

    return result
