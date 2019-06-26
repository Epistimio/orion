import warnings
import datetime
import random

from collections import defaultdict
from track.persistence import get_protocol
from track.structure import Trial as TrackTrial, CustomStatus, Status as TrackStatus


_status = [
    CustomStatus('new', TrackStatus.GroupCreated + 1),
    CustomStatus('reserved', TrackStatus.GroupCreated + 2),

    CustomStatus('suspended', TrackStatus.FinishedGroup + 1),
    CustomStatus('completed', TrackStatus.FinishedGroup + 2),

    CustomStatus('interrupted', TrackStatus.ErrorGroup + 1),
    CustomStatus('broken', TrackStatus.ErrorGroup + 3)
]

_status_dict = {
    s.name: s for s in _status
}


def get_track_status(val):
    return _status.get(val)


class TrialAdapter:
    def __init__(self, storage_trial, orion_trial):
        self.storage = storage_trial
        self.memory = orion_trial


class TrackProtocol:
    def __init__(self, experiement, uri=None):
        self.experiment = experiement
        self.uri = uri
        self.protocol = get_protocol(uri)

    def refresh(self):
        self.protocol = get_protocol(self.uri)

    def create_trial(self, trial):
        self.refresh()

        # self.experiment.register_trial(trial)
        metadata = dict()
        metadata['params_types'] = {p.name: p.type for p in trial.params}
        metadata['submit_time'] = trial.submit_time
        metadata['start_time'] = trial.start_time
        metadata['end_time'] = trial.end_time
        metadata['worker'] = trial.worker
        metadata['metric_types'] = {p.name: p.type for p in trial.results}

        metrics = defaultdict(list)
        for p in trial.results:
            metrics[p.name].append(p.value)

        backend_trial = self.protocol.new_trial(TrackTrial(
            name=trial.full_name,
            _hash=trial.hash_name,
            status=get_track_status(trial.status),
            params={p.name: p.value for p in trial.params},
            metrics=metrics,
            #group_id=trial.experiment
        ))
        self.protocol.commit()
        return TrialAdapter(backend_trial, trial)

    def register_trial(self, trial):
        warnings.warn("deprecated", DeprecationWarning)
        return self.create_trial(trial)

    def fetch_trials(self, query):
        return []

    # ----
    def reserve_trial(self, *args, **kwargs):
        warnings.warn("deprecated", DeprecationWarning)
        return self.select_trial(*args, **kwargs)

    def select_trial(self, *args, **kwargs):
        self.refresh()
        query = dict(
            project_id=self._id,
            status={'$in': ['new', 'suspended', 'interrupted']}
        )

        new_trials = self.fetch_trials(query)

        if not new_trials:
            return None

        selected_trial = random.sample(new_trials, 1)[0]

        # if new add start time
        if selected_trial.status.name == 'new':
            self.protocol.log_trial_metadata(
                trial=selected_trial,
                start_time=datetime.datetime.utcnow()
            )

        # update status to reserved
        self.protocol.set_trial_status(
            trial=selected_trial,
            status=get_track_status('reserved')
        )

        self.protocol.commit()
        return selected_trial
