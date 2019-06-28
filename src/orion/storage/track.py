import warnings
import datetime
import random

from orion.storage.base import BaseStorageProtocol
from orion.core.worker.experiment import Experiment as OrionExp
from orion.core.worker.trial import Trial as OrionTrial

from collections import defaultdict
from track.persistence import get_protocol
from track.structure import Trial as TrackTrial, CustomStatus, Status as TrackStatus, TrialGroup, Project


_status = [
    CustomStatus('new', TrackStatus.CreatedGroup.value + 1),
    CustomStatus('reserved', TrackStatus.CreatedGroup.value + 2),

    CustomStatus('suspended', TrackStatus.FinishedGroup.value + 1),
    CustomStatus('completed', TrackStatus.FinishedGroup.value + 2),

    CustomStatus('interrupted', TrackStatus.ErrorGroup.value + 1),
    CustomStatus('broken', TrackStatus.ErrorGroup.value + 3)
]

_status_dict = {
    s.name: s for s in _status
}
_status_dict['completed'] = TrackStatus.Completed


def get_track_status(val):
    return _status_dict.get(val)


def convert_track_status(status):
    return status.name.lower()


class TrialAdapter:
    def __init__(self, storage_trial, orion_trial=None, objective=None):
        self.storage = storage_trial
        self.memory = orion_trial
        self.session_group = None
        self._params = None
        self.objective_key = objective
        self.objectives_values = None
        self._results = []

    @property
    def id(self):
        return self.storage.uid

    @property
    def params(self):
        if self.memory is not None:
            return self.memory.params

        if self._params is None:
            types = self.storage.metadata['params_types']
            params = self.storage.parameters

            self._params = [
                OrionTrial.Param(name=name, value=params.get(name), type=vtype)
                for name, vtype in types.items()
            ]

        return self._params

    @property
    def status(self):
        if self.memory is not None:
            return self.memory.status

        return convert_track_status(self.storage.status)

    @status.setter
    def status(self, value):
        pass

    def to_dict(self):
        import copy

        if self.memory is not None:
            return self.memory.to_dict()

        trial = copy.deepcopy(self.storage.metadata)
        trial.update({
            'results': self.storage.metrics,
            'params': self.storage.parameters,
            '_id': self.storage.uid,
        })

        return trial

    @property
    def lie(self):
        # we do not lie like Orion does
        return None

    @property
    def objective(self):
        if self.objective_key is None:
            raise RuntimeError('not objective was defined!')

        if self.objectives_values is None:
            self.objectives_values = []

            for k, v in self.storage.metrics[self.objective_key].items():
                self.objectives_values.append((int(k), v))

            self.objectives_values.sort(key=lambda x: x[0])

        return OrionTrial.Result(name=self.objective_key, value=self.objectives_values[-1], type='objective')

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @property
    def gradient(self):
        return None

    @property
    def parents(self):
        return []


class TrackProtocol(BaseStorageProtocol):
    def __init__(self, experiment, uri=None, objective='epoch_loss'):
        super(TrackProtocol, self).__init__()

        self.experiment = experiment
        self.uri = uri
        self.protocol = get_protocol(uri)
        self.session_group = None
        self.project = None
        self.objective = objective

        self.create_session(self.experiment)

    def refresh(self):
        self.protocol = get_protocol(self.uri)

    # In Track we call an experiment a Session
    def create_session(self, experiment: OrionExp):
        import uuid

        project = Project(name='orion_test')
        self.project = self.protocol.get_project(project)

        if self.project is None:
            self.project = self.protocol.new_project(project)

        self.session_group = self.protocol.new_trial_group(TrialGroup(
            name=str(uuid.uuid4()),
            project_id=self.project.uid
        ))

        self.session_group.tags['trials_completed'] = 0
        self.session_group.tags['start_time'] = str(datetime.datetime.utcnow())

        self.protocol.commit()

    def create_trial(self, trial):
        self.refresh()

        stamp = datetime.datetime.utcnow()
        trial.status = 'new'
        trial.submit_time = stamp

        # self.experiment.register_trial(trial)
        metadata = dict()
        metadata['params_types'] = {p.name: p.type for p in trial. params}
        metadata['submit_time'] = str(trial.submit_time)
        metadata['end_time'] = str(trial.end_time)
        metadata['worker'] = trial.worker
        metadata['metric_types'] = {p.name: p.type for p in trial.results}

        metrics = defaultdict(list)
        for p in trial.results:
            metrics[p.name].append(p.value)

        backend_trial = self.protocol.new_trial(TrackTrial(
            _hash=trial.hash_name,
            status=get_track_status('new'),
            parameters={p.name: p.value for p in trial.params},
            metadata=metadata,
            metrics=metrics,
            project_id=self.project.uid,
            group_id=self.session_group.uid
        ))

        trial.experiment = backend_trial.uid
        self.protocol.commit()
        return TrialAdapter(backend_trial, trial, objective=self.objective)

    def register_trial(self, trial):
        warnings.warn("deprecated", DeprecationWarning)
        return self.create_trial(trial)

    def fetch_trials(self, query):
        self.refresh()
        return self.protocol.fetch_trials(query)

    def get_trial(self, uid):
        self.refresh()
        return TrialAdapter(self.protocol.fetch_trials([('uid', uid)])[0], objective=self.objective)

    def fetch_completed_trials(self):
        query = [
            ('group_id'  , self.session_group.uid),
            ('project_id', self.project.uid),
            ('status'    , {
                '$in': [
                    get_track_status('completed'),
                    TrackStatus.Completed
                ]
            }),
            # end_time={'$gte': self._last_fetched}
        ]
        last_fetched = datetime.datetime.utcnow()
        data = self.fetch_trials(query)
        return [TrialAdapter(t, objective=self.objective) for t in data]

    # ----
    def reserve_trial(self, *args, **kwargs):
        warnings.warn("deprecated", DeprecationWarning)
        return self.select_trial(*args, **kwargs)

    def select_trial(self, *args, **kwargs):
        self.refresh()

        query = [
            ('group_id'  , self.session_group.uid),
            ('project_id', self.project.uid),
            ('status'    , {'$in': [
                get_track_status('new'),
                get_track_status('suspended'),
                get_track_status('interrupted')
            ]})
        ]

        new_trials = self.fetch_trials(query)

        if not new_trials:
            return None

        selected_trial = random.sample(new_trials, 1)[0]

        # if new add start time
        if selected_trial.status.name == 'new':
            self.protocol.log_trial_metadata(
                trial=selected_trial,
                start_time=str(datetime.datetime.utcnow())
            )

        # update status to reserved
        self.protocol.set_trial_status(
            trial=selected_trial,
            status=get_track_status('reserved')
        )

        self.protocol.commit()
        return TrialAdapter(selected_trial, objective=self.objective)

    def is_done(self, experiment):
        query = [
            ('group_id', self.session_group.uid),
            ('project_id', self.project.uid),
            ('status', {
                '$in': [
                    get_track_status('completed'),
                    TrackStatus.Completed
                ]
            }),
        ]

        num_completed_trials = len(self.fetch_trials(query))
        done = ((num_completed_trials >= experiment.max_trials) or
                (experiment._init_done and experiment.algorithms.is_done))

        if done:
            self.session_group.tags['finish_time'] = str(datetime.datetime.utcnow())

        return done

    def push_completed_trial(self, trial):
        self.refresh()
        trial = self.get_trial(trial.id)
        self.session_group = self.protocol.get_trial_group(TrialGroup(_uid=self.session_group.uid))

        curent_obj = trial.objective.value
        try:
            if len(curent_obj) > 0:
                curent_obj = curent_obj[-1]
        except:
            pass

        best_obj = self.session_group.tags.get('best_evaluation')

        if best_obj is None or best_obj > curent_obj:
            self.session_group.tags['best_trials_id'] = trial.id
            self.session_group.tags['best_evaluation'] = curent_obj

        self.session_group.tags['trials_completed'] += 1
        self.protocol.commit()

    def mark_as_broken(self, trial):
        self.protocol.set_trial_status(TrackTrial(hash=trial.id), get_track_status('broken'))

    def get_stats(self):
        return self.session_group.tags
