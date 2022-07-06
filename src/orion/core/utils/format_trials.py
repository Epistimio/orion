"""
Utility functions for formatting data
=====================================

Conversion functions between various data types used in framework's ecosystem.

"""

from orion.core.utils.flatten import flatten
from orion.core.worker.trial import Trial


def trial_to_tuple(trial, space):
    """Extract a parameter tuple from a `orion.core.worker.trial.Trial`.

    The order within the tuple is dictated by the defined
    `orion.algo.space.Space` object.
    """
    params = flatten(trial.params)
    trial_keys = set(params.keys())
    space_keys = set(space.keys())
    if trial_keys != space_keys:
        raise ValueError(
            f"The trial {trial.id} has wrong params:\n"
            f"Trial params: {sorted(trial_keys)}\n"
            f"Space dims: {sorted(space_keys)}"
        )
    return tuple(params[name] for name in space.keys())


def dict_to_trial(data, space):
    """Create a `orion.core.worker.trial.Trial` object from `data`,
    filling only parameter information from `data`.

    :param data: A dict representing a sample point from `space`.
    :param space: Definition of problem's domain.
    :type space: `orion.algo.space.Space`
    """
    data = flatten(data)
    params = []
    for name, dim in space.items():
        if name not in data and dim.default_value is dim.NO_DEFAULT_VALUE:
            raise ValueError(
                f"Dimension {name} not specified and does not have a default value."
            )
        value = data.get(name, dim.default_value)

        params.append(dict(name=dim.name, type=dim.type, value=value))

    trial = Trial(params=params)

    if trial not in space:
        error_msg = f"Parameters values {trial.params} are outside of space {space}"
        raise ValueError(error_msg)

    return trial


def tuple_to_trial(data, space, status="new"):
    """Create a `orion.core.worker.trial.Trial` object from `data`.

    Parameters
    ----------
    data: tuple
        A tuple representing a sample point from `space`.
    space: `orion.algo.space.Space`
        Definition of problem's domain.
    status: str, optional
        Status of the trial. One of ``orion.core.worker.trial.Trial.allowed_stati``.

    Returns
    -------
    A trial object `orion.core.worker.trial.Trial`.
    """
    if len(data) != len(space):
        raise ValueError(
            f"Data point is not compatible with search space:\ndata: {data}\nspace: {space}"
        )

    params = []
    for i, dim in enumerate(space.values()):
        params.append(dict(name=dim.name, type=dim.type, value=data[i]))

    return Trial(params=params, status=status)


def get_trial_results(trial):
    """Format results from a `orion.core.worker.trial.Trial` using standard structures."""
    results = {}

    lie = trial.lie
    objective = trial.objective

    if lie:
        results["objective"] = lie.value
    elif objective:
        results["objective"] = objective.value
    else:
        results["objective"] = None

    results["constraint"] = [
        result.value for result in trial.results if result.type == "constraint"
    ]
    grad = trial.gradient
    results["gradient"] = tuple(grad.value) if grad else None

    return results


def standard_param_name(name):
    """Convert parameter name to namespace format"""
    return name.lstrip("/").lstrip("-").replace("-", "_")
