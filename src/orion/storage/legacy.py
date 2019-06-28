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
        """create a trial to be ran in the future"""
        self.experiment.register_trial(trial)

    def register_trial(self, trial):
        """legacy function @see create_trial"""
        warnings.warn("deprecated", DeprecationWarning)
        return self.create_trial(trial)

    def select_trial(self, *args, **kwargs):
        """select pending trials that should be ran next"""
        return self.experiment.reserve_trial(*args, **kwargs)

    def reserve_trial(self, *args, **kwargs):
        """legacy function mark a trial as reserved since it will be ran shortly"""
        warnings.warn("deprecated", DeprecationWarning)
        return self.select_trial(*args, **kwargs)

    def fetch_completed_trials(self):
        """fetch all the trials that are marked as completed"""
        return self.experiment.fetch_completed_trials()

    def is_done(self, experiment):
        """check if we have reached the maximum number of completed trials"""
        return self.experiment.is_done

    def push_completed_trial(self, trial):
        """make the trial as complete and update experiment statistics"""
        self.experiment.push_completed_trial(trial)

    def mark_as_broken(self, trial):
        """mark the trial as broken to avoid retrying a failing trial"""
        trial.status = 'broken'
        Database().write(
            'trials',
            trial.to_dict(),
            query={
                '_id': trial.id
            }
        )

    def get_stats(self):
        """return the stats from the experiment"""
        return self.experiment.stats

    def update_trial(self, trial, results_file=None, **kwargs):
        """read the results from the trial and append it to the trial object"""
        results = self.converter.parse(results_file.name)

        trial.results = [
            Trial.Result(
                name=res['name'],
                type=res['type'],
                value=res['value']) for res in results
        ]
        return trial

    def get_trial(self, uid):
        """fetch the trial from the database """
        return Trial(**Database().read('trials', {'_id': uid})[0])
