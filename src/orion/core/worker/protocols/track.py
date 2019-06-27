import warnings
import datetime
import random

from orion.core.worker.experiment import Experiment as OrionExp

from collections import defaultdict
from track.persistence import get_protocol
from track.structure import Trial as TrackTrial, CustomStatus, Status as TrackStatus, TrialGroup, Project


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
        self.session_group = None


class TrackProtocol:
    def __init__(self, experiement, uri=None):
        self.experiment = experiement
        self.uri = uri
        self.protocol = get_protocol(uri)
        self.session_group = None
        self.project = None

        self.create_session(self.experiment)

    def refresh(self):
        self.protocol = get_protocol(self.uri)

    # In Track we call an experiment a Session
    def create_session(self, experiment: OrionExp):
        self.project = self.protocol.get_project(Project(
            name=experiment.name
        ))

        self.session_group = self.protocol.get_trial_group(TrialGroup(
            name=f'session_{experiment.name}',
            project_id=self.project.uid
        ))

    def create_trial(self, trial):
        self.refresh()

        # self.experiment.register_trial(trial)
        metadata = dict()
        metadata['params_types'] = {p.name: p.type for p in trial. params}
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
            group_id=self.session_group.uid
        ))
        self.protocol.commit()
        return TrialAdapter(backend_trial, trial)

    def register_trial(self, trial):
        warnings.warn("deprecated", DeprecationWarning)
        return self.create_trial(trial)

    def fetch_trials(self, query):
        return self.protocol.fetch_trials(query)

    # ----
    def reserve_trial(self, *args, **kwargs):
        warnings.warn("deprecated", DeprecationWarning)
        return self.select_trial(*args, **kwargs)

    def select_trial(self, *args, **kwargs):
        self.refresh()
        query = dict(
            project_id=self.project.uid,
            status={'$in': [
                get_track_status('new'),
                get_track_status('suspended'),
                get_track_status('interrupted')
            ]}
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
