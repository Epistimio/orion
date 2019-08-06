# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.legacy -- Track Storage Protocol
====================================================

.. module:: base
   :platform: Unix
   :synopsis: Implement a storage protocol to allow Orion to use track as a storage method

"""

from collections import defaultdict
import copy
import datetime
import logging
import uuid

try:
    from track.client import TrackClient
    from track.persistence.utils import parse_uri
    from track.serialization import to_json
    from track.structure import CustomStatus, Status as TrackStatus
    from track.structure import Project, Trial as TrackTrial, TrialGroup

    HAS_TRACK = True
except ImportError:
    HAS_TRACK = False

from orion.core.worker.trial import Trial as OrionTrial
from orion.storage.base import BaseStorageProtocol


log = logging.getLogger(__name__)


if HAS_TRACK:
    _status = [
        CustomStatus('new', TrackStatus.CreatedGroup.value + 1),
        CustomStatus('reserved', TrackStatus.CreatedGroup.value + 2),
    ]

    _status_dict = {
        s.name: s for s in _status
    }
    _status_dict['completed'] = TrackStatus.Completed
    _status_dict['interrupted'] = TrackStatus.Interrupted
    _status_dict['broken'] = TrackStatus.Broken
    _status_dict['suspended'] = TrackStatus.Suspended


def get_track_status(val):
    """Convert orion status to track status"""
    return _status_dict.get(val)


def convert_track_status(status):
    """Convert track status to orion status"""
    return status.name.lower()


def remove_leading_slash(name):
    """Remove leading slash"""
    # if name[0] == '/':
    #     return name[1:]
    # return name
    return name


def add_leading_slash(name):
    """Add leading slash"""
    # if name[0] == '/':
    #     return name
    # return '/' + name
    return name


class TrialAdapter:
    """Mock Trial, see `~orion.core.worker.trial.Trial`

    Parameters
    ----------
    storage_trial
        Track trial object

    orion_trial
        Orion trial object

    objective: str
        objective key

    """

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
        ret = "TrialAdapter(uid={3}, experiment={0}, status={1}, params={2})".format(
            repr(self.experiment[:10]), repr(self.status), param_rep, self.storage.uid)
        return ret

    __repr__ = __str__

    @property
    def experiment(self):
        """See `~orion.core.worker.trial.Trial`"""
        if self.memory is not None:
            return self.memory.experiment
        return self.storage.group_id

    @property
    def hearbeat(self):
        """See `~orion.core.worker.trial.Trial`"""
        return datetime.datetime.utcfromtimestamp(self.storage.metadata.get('heartbeat', 0))

    @property
    def id(self):
        """See `~orion.core.worker.trial.Trial`"""
        return self.storage.uid

    @property
    def params(self):
        """See `~orion.core.worker.trial.Trial`"""
        if self.memory is not None:
            return self.memory.params

        types = self.storage.metadata['params_types']
        params = self.storage.parameters

        self._params = [
            OrionTrial.Param(name=add_leading_slash(name), value=params.get(name), type=vtype)
            for name, vtype in types.items()
        ]

        return self._params

    @property
    def status(self):
        """See `~orion.core.worker.trial.Trial`"""
        if self.memory is not None:
            return self.memory.status

        return convert_track_status(self.storage.status)

    @status.setter
    def status(self, value):
        """See `~orion.core.worker.trial.Trial`"""
        pass

    def to_dict(self):
        """See `~orion.core.worker.trial.Trial`"""
        trial = copy.deepcopy(self.storage.metadata)
        trial.update({
            'results': self.storage.metrics,
            'params': self.storage.parameters,
            '_id': self.storage.uid,
        })

        return trial

    @property
    def lie(self):
        """See `~orion.core.worker.trial.Trial`"""
        # we do not lie like Orion does
        return None

    @property
    def objective(self):
        """See `~orion.core.worker.trial.Trial`"""
        def result(val):
            return OrionTrial.Result(name=self.objective_key, value=val, type='objective')

        if self.objective_key is None:
            raise RuntimeError('no objective key was defined!')

        self.objectives_values = []

        data = self.storage.metrics.get(self.objective_key)
        if data is None:
            return None

        # objective was pushed without step data (already sorted)
        if isinstance(data, list):
            self.objectives_values = data
            return result(self.objectives_values[-1])

        # objective was pushed with step data
        elif isinstance(data, dict):
            for k, v in self.storage.metrics[self.objective_key].items():
                self.objectives_values.append((int(k), v))

            self.objectives_values.sort(key=lambda x: x[0])
            return result(self.objectives_values[-1][1])

        return None

    @property
    def results(self):
        """See `~orion.core.worker.trial.Trial`"""
        return self._results

    @results.setter
    def results(self, value):
        """See `~orion.core.worker.trial.Trial`"""
        self._results = value

    @property
    def gradient(self):
        """See `~orion.core.worker.trial.Trial`"""
        return None

    @property
    def parents(self):
        """See `~orion.core.worker.trial.Trial`"""
        return []

    @property
    def submit_time(self):
        """See `~orion.core.worker.trial.Trial`"""
        return datetime.datetime.utcfromtimestamp(self.storage.metadata.get('submit_time'))

    @property
    def end_time(self):
        """See `~orion.core.worker.trial.Trial`"""
        return datetime.datetime.utcfromtimestamp(self.storage.metadata.get('end_time'))

    @end_time.setter
    def end_time(self, value):
        """See `~orion.core.worker.trial.Trial`"""
        self.storage.metadata['end_time'] = value


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
        self.options = parse_uri(uri)['query']

        self.client = TrackClient(uri)
        self.backend = self.client.protocol
        self.project = None
        self.group = None
        self.current_trial = None

        self.objective = self.options.get('objective')
        assert self.objective is not None, 'An objective should be defined!'

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""
        if self.project is None:
            self.project = self.backend.get_project(Project(name=config['name']))

            if self.project is None:
                self.project = self.backend.new_project(Project(name=config['name']))

        self.group = self.backend.new_trial_group(
            TrialGroup(
                name=str(uuid.uuid4()),
                project_id=self.project.uid,
                metadata=to_json(config)
            )
        )

        config['_id'] = self.group.uid
        return config

    def update_experiment(self, experiment, where=None, **kwargs):
        """Update the fields of a given trials

        Parameters
        ----------
        experiment: Experiment
            Experiment object to update

        where: Optional[dict]
            constraint experiment must respect for the update to take place

        **kwargs: dict
            a dictionary of fields to update

        Returns
        -------
        returns true if the underlying storage was updated

        """
        raise RuntimeError('You should not update a track experiment')

    def fetch_experiments(self, query):
        """Fetch all experiments that match the query"""
        return self.backend.fetch_groups(query)

    def register_trial(self, trial):
        """Create a new trial to be executed"""
        stamp = datetime.datetime.utcnow()
        trial.status = 'new'
        trial.submit_time = stamp

        metadata = dict()
        metadata['params_types'] = {remove_leading_slash(p.name): p.type for p in trial.params}
        metadata['submit_time'] = to_json(trial.submit_time)
        metadata['end_time'] = to_json(trial.end_time)
        metadata['worker'] = trial.worker
        metadata['metric_types'] = {remove_leading_slash(p.name): p.type for p in trial.results}
        metadata['metric_types'][self.objective] = 'objective'
        metadata['heartbeat'] = stamp

        metrics = defaultdict(list)
        for p in trial.results:
            metrics[p.name] = [p.value]

        self.current_trial = self.backend.new_trial(TrackTrial(
            _hash=trial.hash_name,
            status=get_track_status('new'),
            project_id=self.project.uid,
            group_id=self.group.uid,
            parameters={p.name: p.value for p in trial.params},
            metadata=metadata,
            metrics=metrics
        ))
        return TrialAdapter(self.current_trial, objective=self.objective)

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
        def sort_key(item):
            submit_time = item.submit_time
            if submit_time is None:
                return 0
            return submit_time

        query = to_json(query)

        new_query = {}
        for k, v in query.items():
            if k == 'experiment':
                new_query['group_id'] = v

            elif k == 'heartbeat':
                new_query['metadata.heartbeat'] = v

            elif k == '_id':
                new_query['uid'] = v

            elif k == 'end_time':
                new_query['metadata.end_time'] = v

            elif k == 'status' and isinstance(v, str):
                new_query['status'] = get_track_status(v)

            else:
                new_query[k] = v

        trials = [
            TrialAdapter(t, objective=self.objective) for t in self.backend.fetch_trials(new_query)
        ]
        trials.sort(key=sort_key)
        return trials

    _ignore_updates_for = {'results', 'params', '_id'}

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
        if isinstance(trial, TrialAdapter):
            trial = trial.storage

        for key, value in kwargs.items():
            if key == 'status':
                self.backend.set_trial_status(trial, get_track_status(value))
            elif key in self._ignore_updates_for:
                continue
            else:
                pair = {key: to_json(value)}
                self.backend.log_trial_metadata(trial, **pair)

        return True

    def retrieve_result(self, trial, *args, **kwargs):
        """Fetch the result from a given medium (file, db, socket, etc..) for a given trial and
        insert it into the trial object
        """
        if isinstance(trial, TrialAdapter):
            trial = trial.storage

        refreshed_trial = self.backend.get_trial(trial)[0]
        new_trial = TrialAdapter(refreshed_trial, objective=self.objective)

        assert new_trial.objective is not None, 'Trial should have returned an objective value!'

        log.info("trial objective is (%s: %s)", self.objective, new_trial.objective.value)
        return new_trial

    def fetch_pending_trials(self, experiment):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_pending_trials`"""

        query = dict(
            group_id=experiment._id,
            status={'$in': [
                'new',
                'suspended',
                'interrupted'
            ]}
        )
        return self.fetch_trials(query)
