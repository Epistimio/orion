import warnings


class DebugProtocol:
    def __init__(self, experiment, uri=None):
        self.experiment = experiment

    def create_trial(self, trial):
        self.experiment.register_trial(trial)

    def register_trial(self, trial):
        warnings.warn("deprecated", DeprecationWarning)
        return self.create_trial(trial)

    def select_trial(self, *args, **kwargs):
        return self.experiment.reserve_trial(*args, **kwargs)

    def reserve_trial(self, *args, **kwargs):
        warnings.warn("deprecated", DeprecationWarning)
        return self.select_trial(*args, **kwargs)

    def fetch_completed_trials(self):
        return self.experiment.fetch_completed_trials()

