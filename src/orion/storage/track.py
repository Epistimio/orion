# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.legacy -- Track Storage Protocol
====================================================

.. module:: base
   :platform: Unix
   :synopsis: Implement a storage protocol to allow Orion to use track as a storage method

"""
import uuid
import datetime
from collections import defaultdict

from orion.core.worker.trial import Trial as OrionTrial
from orion.storage.base import BaseStorageProtocol

from track.serialization import to_json
from track.client import TrackClient
from track.structure import Trial as TrackTrial, TrialGroup, Project
from track.structure import CustomStatus, Status as TrackStatus


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

    def _repr_values(self, values, sep=','):
        """Represent with a string the given values."""
        return

    def __str__(self):
        """Represent partially with a string."""
        param_rep = ','.join(map(lambda value: "{0.name}:{0.value}".format(value), self.params))
        ret = "TrialAdapter(experiment={0}, status={1}, params={2})".format(
            repr(self.experiment), repr(self.status), param_rep)
        return ret

    __repr__ = __str__

    @property
    def experiment(self):
        if self.memory is not None:
            return self.memory.experiment
        return self.storage.group_id

    @property
    def hearbeat(self):
        return datetime.datetime.utcfromtimestamp(self.storage.metadata.get('heartbeat', 0))

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

    @property
    def submit_time(self):
        return self.storage.metadata.get('submit_time')


class Track(BaseStorageProtocol):
    """Implement a generic protocol to allow Orion to communicate using
    different storage backend

    Parameters
    ----------
    uri: str
        Track backend to use for storage; the format is as follow
         `protocol://[username:password@]host1[:port1][,...hostN[:portN]]][/[database][?options]]`
    """

    def __init__(self, uri):
        self.uri = uri
        self.client = TrackClient(uri)
        self.backend = self.client.protocol
        self.project = None
        self.group = None
        self.current_trial = None

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""

        if self.project is None:
            self.project = self.backend.get_project(Project(name=config['name']))

            if self.project is None:
                self.project = self.backend.new_project(Project(name=config['name']))

        self.group = self.backend.new_trial_group(TrialGroup(name=str(uuid.uuid4()), project_id=self.project.uid))

        config['_id'] = self.group.uid
        return config

    def update_experiment(self, experiment, where=None, **kwargs):
        """Update a the fields of a given trials

        Parameters
        ----------
        experiment: Experiment
            Experiment object to update

        where: Optional[dict]
            constraint experiment must respect

        **kwargs: dict
            a dictionary of fields to update

        Returns
        -------
        returns true if the underlying storage was updated

        """
        pass

    def fetch_experiments(self, query):
        """Fetch all experiments that match the query"""
        return self.backend.fetch_groups(query)

    def register_trial(self, trial):
        """Create a new trial to be executed"""
        stamp = datetime.datetime.utcnow()
        trial.status = 'new'
        trial.submit_time = stamp

        metadata = dict()
        metadata['params_types'] = {p.name: p.type for p in trial. params}
        metadata['submit_time'] = to_json(trial.submit_time)
        metadata['end_time'] = to_json(trial.end_time)
        metadata['worker'] = trial.worker
        metadata['metric_types'] = {p.name: p.type for p in trial.results}

        metrics = defaultdict(list)
        for p in trial.results:
            metrics[p.name] = p.value

        self.current_trial = self.backend.new_trial(TrackTrial(
            _hash=trial.hash_name,
            status=get_track_status('new'),
            project_id=self.project.uid,
            group_id=self.group.uid,
            parameters={p.name: p.value for p in trial.params},
            metadata=metadata,
            metrics=metrics
        ))
        return TrialAdapter(self.current_trial)

    def register_lie(self, trial):
        """Register a *fake* trial created by the strategist.

        The main difference between fake trial and original ones is the addition of a fake objective
        result, and status being set to completed. The id of the fake trial is different than the id
        of the original trial, but the original id can be computed using the hashcode on parameters
        of the fake trial. See mod:`orion.core.worker.strategy` for more information and the
        Strategist object and generation of fake trials.

        Parameters
        ----------
        trial: `Trial` object
            Fake trial to register in the database

        """
        pass

    def fetch_trials(self, query, *args, **kwargs):
        """Fetch all the trials that match the query"""
        query = to_json(query)
        new_query = {}
        for k, v in query.items():
            if k == 'experiment':
                new_query['group_id'] = v
            elif k == 'heartbeat':
                new_query['metadata.heartbeat'] = v
            elif k == '_id':
                new_query['uid'] = v
            else:
                new_query[k] = v

        results = [TrialAdapter(t) for t in self.backend.fetch_trials(new_query)]
        return results

    def update_trial(self, trial, where=None, **kwargs):
        """Update the fields of a given trials

        Parameters
        ----------
        trial: Trial
            Trial object to update

        where: Optional[dict]
            constraint trial must respect

        kwargs: dict
            a dictionary of fields to update

        Returns
        -------
        returns true if the underlying storage was updated

        """
        try:
            if isinstance(trial, TrialAdapter):
                trial = trial.storage

            for key, value in kwargs.items():
                if key == 'status':
                    self.backend.set_trial_status(trial, get_track_status(value))
                else:
                    pair = {key: to_json(value)}
                    self.backend.log_trial_metadata(trial, **pair)

            return True

        except RuntimeError:
            return False

    def retrieve_result(self, trial, *args, **kwargs):
        """Fetch the result from a given medium (file, db, socket, etc..) for a given trial and
        insert it into the trial object
        """
        if isinstance(trial, TrialAdapter):
            trial = trial.storage

        self.client = TrackClient(self.uri)
        self.backend = self.client.protocol

        refreshed_trial = self.backend.get_trial(trial)
        print(refreshed_trial)
        return TrialAdapter(refreshed_trial)

    def fetch_pending_trials(self, experiment):
        """Fetch trials that have not run yet"""
        query = dict(
            group_id=experiment._id,
            status={'$in': [
                'new',
                'suspended',
                'interrupted'
            ]}
        )
        return self.fetch_trials(query)
