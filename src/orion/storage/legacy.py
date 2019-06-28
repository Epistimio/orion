import warnings

from orion.storage.base import BaseStorageProtocol
from orion.core.io.database import Database
from orion.core.io.convert import JSONConverter
from orion.core.worker.trial import Trial


class Legacy(BaseStorageProtocol):
    def __init__(self, experiment, uri=None):
        self.experiment = experiment
        self.converter = JSONConverter()

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

    def update_trial(self, trial, results_file=None, **kwargs):
        results = self.converter.parse(results_file.name)

        trial.results = [
            Trial.Result(
                name=res['name'],
                type=res['type'],
                value=res['value']) for res in results
        ]
        return trial

    def get_trial(self, uid):
        return Trial(**Database().read('trials', {'_id': uid})[0])
