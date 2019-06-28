import warnings
from orion.core.io.database import Database


class LegacyProtocol:
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

    def is_done(self, experiment):
        return self.experiment.is_done

    def push_completed_trial(self, trial):
        self.experiment.push_completed_trial(trial)

    def mark_as_broken(self, trial):
        trial.status = 'broken'
        Database().write(
            'trials',
            trial.to_dict(),
            query={
                '_id': trial.id
            }
        )

    def get_stats(self):
        return self.experiment.stats
