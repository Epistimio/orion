import contextlib
import copy

from orion.client import build_experiment, get_experiment


@contextlib.contextmanager
def disable_duplication(monkeypatch):
    def stub(self):
        pass

    with monkeypatch.context() as m:
        m.setattr(
            "orion.core.worker.experiment.Experiment.duplicate_pending_trials", stub
        )

        yield


def generate_trials(exp, trials):
    """Generate trials for each item in trials.

    Items of trials can be either dictionary of valid hyperparameters based on exp.space and status
    or `None`.

    If status not provided, 'new' is used by default.

    For items that are `None`, trials are suggested with exp.suggest().
    """
    for trial_config in trials:
        trial_config = copy.deepcopy(trial_config)
        status = trial_config.pop("status", None) if trial_config else None
        if trial_config:
            trial = exp.insert(params=trial_config)
        else:
            with exp.suggest() as trial:
                # Releases suggested trial when leaving with-clause.
                pass

        if status is not None:
            exp._experiment._storage.set_trial_status(
                trial,
                status,
                heartbeat=trial.submit_time if status == "reserved" else None,
            )


def build_root_experiment(space=None, trials=None):
    """Build a root experiment and generate trials."""
    if space is None:
        space = {"x": "uniform(0, 100)", "y": "uniform(0, 100)", "z": "uniform(0, 100)"}
    if trials is None:
        trials = [{"x": i, "y": i * 2, "z": i ** 2} for i in range(4)]

    root = build_experiment(name="root", max_trials=len(trials), space=space)

    generate_trials(root, trials)


def build_child_experiment(space=None, trials=None, name="child", parent="root"):
    """Build a child experiment by branching from `parent` and generate trials."""
    if trials is None:
        trials = [None for i in range(6)]

    max_trials = get_experiment(parent).max_trials + len(trials)

    child = build_experiment(
        name=name,
        space=space,
        max_trials=max_trials,
        branching={"branch_from": parent, "enable": True},
    )
    assert child.name == name
    assert child.version == 1

    generate_trials(child, trials)


def build_grand_child_experiment(space=None, trials=None):
    """Build a grand-child experiment by branching from `child` and generate trials."""
    if trials is None:
        trials = [None for i in range(5)]

    build_child_experiment(
        space=space, trials=trials, name="grand-child", parent="child"
    )
