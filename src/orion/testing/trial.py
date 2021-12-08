import string

from orion.core.worker.trial import Trial


def create_trial(point, names=None, types=None, results=None):
    if names is None:
        names = string.ascii_lowercase[: len(point)]

    if types is None:
        types = ["real"] * len(point)

    return Trial(
        params=[
            {"name": name, "value": value, "type": param_type}
            for (name, value, param_type) in zip(names, point, types)
        ],
        results=[
            {"name": name, "type": name, "value": value}
            for name, value in results.items()
        ],
    )


def compare_trials(trials, other_trials):
    assert [t.params for t in trials] == [t.params for t in other_trials]
