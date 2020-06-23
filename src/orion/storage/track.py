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
import hashlib
import logging
import sys
import warnings

from orion.core.io.database import DuplicateKeyError
from orion.storage.base import BaseStorageProtocol, FailedUpdate, MissingArguments

log = logging.getLogger(__name__)


# TODO: Remove this when factory is reworked
class Track:    # noqa: F811
    """Forward declaration because of a weird factory bug where Track is not found"""

    def __init__(self, uri):
        assert False, 'This should not be called'


HAS_TRACK = False
REASON = None
try:
    from track.client import TrackClient
    from track.persistence.utils import parse_uri
    from track.serialization import to_json
    from track.structure import CustomStatus, Status as TrackStatus
    from track.structure import Project, Trial as TrackTrial, TrialGroup
    from track.persistence.local import ConcurrentWrite

    HAS_TRACK = True
except ImportError:
    REASON = 'Track is not installed'

except SyntaxError:
    major, minor, patch, _, _ = sys.version_info

    if minor < 6:
        REASON = 'Python is too old'
        log.warning('Track does not support python < 3.6!')
    else:
        raise


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


def to_epoch(date):
    """Convert datetime class into seconds since epochs"""
    return (date - datetime.datetime(1970, 1, 1)).total_seconds()


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
        self.storage = copy.deepcopy(storage_trial)
        self.memory = orion_trial
        self.session_group = None
        self.objective_key = objective
        self.objectives_values = None
        self._results = []

    def _repr_values(self, values, sep=','):
        """Represent with a string the given values."""
        return

    def __str__(self):
        """Represent partially with a string."""
        param_rep = ','.join(map(lambda value: "{0.name}:{0.value}".format(value), self._params))
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

        return {param.name: param.value for param in self._params}

    @property
    def _params(self):
        """See `~orion.core.worker.trial.Trial`"""
        from orion.core.worker.trial import Trial as OrionTrial

        if self.memory is not None:
            return self.memory._params

        types = self.storage.metadata['params_types']
        params = self.storage.parameters

        return [
            OrionTrial.Param(name=add_leading_slash(name), value=params.get(name), type=vtype)
            for name, vtype in types.items()
        ]

    @property
    def status(self):
        """See `~orion.core.worker.trial.Trial`"""
        if self.memory is not None:
            return self.memory.status

        return convert_track_status(self.storage.status)

    @status.setter
    def status(self, value):
        """See `~orion.core.worker.trial.Trial`"""
        self.storage.status = get_track_status(value)

        if self.memory is not None:
            self.memory.status = value

    def to_dict(self):
        """See `~orion.core.worker.trial.Trial`"""
        trial = copy.deepcopy(self.storage.metadata)
        trial.update({
            'results': [r.to_dict() for r in self.results],
            'params': [p.to_dict() for p in self._params],
            '_id': self.storage.uid,
            'submit_time': self.submit_time,
            'experiment': self.experiment,
            'status': self.status
        })

        trial.pop('_update_count', 0)
        trial.pop('metric_types', 0)
        trial.pop('params_types')

        return trial

    @property
    def lie(self):
        """See `~orion.core.worker.trial.Trial`"""
        # we do not lie like Orion does
        return None

    @property
    def objective(self):
        """See `~orion.core.worker.trial.Trial`"""
        from orion.core.worker.trial import Trial as OrionTrial

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
        from orion.core.worker.trial import Trial as OrionTrial

        self._results = []

        for k, values in self.storage.metrics.items():
            result_type = 'statistic'
            if k == self.objective_key:
                result_type = 'objective'

            self._results.append(OrionTrial.Result(name=k, type=result_type, value=values[-1]))

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

    @property
    def heartbeat(self):
        """Trial Heartbeat"""
        heartbeat = self.storage.metadata.get('heartbeat')
        if heartbeat:
            return datetime.datetime.utcfromtimestamp(heartbeat)
        return None


def experiment_uid(exp=None, name=None, version=None):
    """Return an experiment uid from its name and version for Track"""
    if name is None:
        name = exp.name

    if version is None:
        version = exp.version

    sha = hashlib.sha256()
    sha.update(name.encode('utf8'))
    sha.update(bytes([version]))
    return sha.hexdigest()


class Track(BaseStorageProtocol):   # noqa: F811
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
        self.lies = dict()
        assert self.objective is not None, 'An objective should be defined!'

    def _get_project(self, name):
        if self.project is None:
            self.project = self.backend.get_project(Project(name=name))

            if self.project is None:
                self.project = self.backend.new_project(Project(name=name))

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""
        self._get_project(config['name'])

        self.group = self.backend.new_trial_group(
            TrialGroup(
                name=experiment_uid(name=config['name'], version=config['version']),
                project_id=self.project.uid,
                metadata=to_json(config)
            )
        )

        if self.group is None:
            raise DuplicateKeyError('Experiment was already created')

        config['_id'] = self.group.uid
        return config

    def update_experiment(self, experiment=None, uid=None, where=None, **kwargs):
        """See :func:`~orion.storage.BaseStorageProtocol.update_experiment`"""
        if uid and experiment:
            assert experiment._id == uid

        if uid is None:
            if experiment is None:
                raise MissingArguments('experiment or uid need to be defined')
            else:
                uid = experiment._id

        self.group = self.backend.fetch_and_update_group({
            '_uid': uid
        }, 'set_group_metadata', **kwargs)

        return self.group

    def fetch_experiments(self, query, selection=None):
        """Fetch all experiments that match the query"""
        new_query = {}
        for k, v in query.items():
            if k == 'name':
                new_query['metadata.name'] = v

            elif k.startswith('metadata'):
                new_query['metadata.{}'.format(k)] = v

            elif k == '_id':
                new_query['_uid'] = v

            else:
                new_query[k] = v

        groups = self.backend.fetch_groups(new_query)

        experiments = []
        for group in groups:
            version = group.metadata.get('version', 0)

            # metadata is experiment config
            exp = group.metadata
            exp.update({
                '_id': group.uid,
                'version': version,
                'name': group.project_id,
            })

            experiments.append(exp)

        return experiments

    def register_trial(self, trial):
        """Create a new trial to be executed"""
        stamp = datetime.datetime.utcnow()
        trial.submit_time = stamp

        metadata = dict()
        # pylint: disable=protected-access
        metadata['params_types'] = {remove_leading_slash(p.name): p.type for p in trial._params}
        metadata['submit_time'] = to_json(trial.submit_time)
        metadata['end_time'] = to_json(trial.end_time)
        metadata['worker'] = trial.worker
        metadata['metric_types'] = {remove_leading_slash(p.name): p.type for p in trial.results}
        metadata['metric_types'][self.objective] = 'objective'
        heartbeat = to_json(trial.heartbeat)
        if heartbeat is None:
            heartbeat = 0
        metadata['heartbeat'] = heartbeat

        metrics = defaultdict(list)
        for p in trial.results:
            metrics[p.name] = [p.value]

        self.current_trial = self.backend.new_trial(TrackTrial(
            _hash=trial.hash_name,
            status=get_track_status(trial.status),
            project_id=self.project.uid,
            group_id=self.group.uid,
            parameters=trial.params,
            metadata=metadata,
            metrics=metrics
        ), auto_increment=False)

        if self.current_trial is None:
            raise DuplicateKeyError('Was not able to register Trial!')

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
        warnings.warn('Track does not persist lies!')

        if trial.id in self.lies:
            raise DuplicateKeyError('Lie already exists')

        self.lies[trial.id] = trial
        return trial

    def _fetch_trials(self, query, *args, **kwargs):
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

    def _update_trial(self, trial, **kwargs):
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
                elif key in self._ignore_updates_for:
                    continue
                else:
                    pair = {key: to_json(value)}
                    self.backend.log_trial_metadata(trial, **pair)

            return True
        except ConcurrentWrite:
            return False

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
        pending_status = ['new', 'suspended', 'interrupted']
        pending_status = [get_track_status(s) for s in pending_status]

        query = dict(
            group_id=experiment.id,
            status={'$in': pending_status}
        )

        return self._fetch_trials(query)

    def set_trial_status(self, trial, status, heartbeat=None):
        """Update the trial status and the heartbeat

        Raises
        ------
        FailedUpdate
            The exception is raised if the status of the trial object
            does not match the status in the database

        """
        result_trial = self.backend.fetch_and_update_trial({
            'uid': trial.id,
            'status': get_track_status(trial.status)
        }, 'set_trial_status', status=get_track_status(status))

        if result_trial is None:
            raise FailedUpdate()

        trial.status = status
        return result_trial

    def fetch_trials(self, experiment=None, uid=None):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_trials`"""
        if uid and experiment:
            assert experiment.id == uid

        if uid is None:
            if experiment is None:
                raise MissingArguments('experiment or uid need to be defined')

            uid = experiment.id

        return self._fetch_trials(dict(group_id=uid))

    def get_trial(self, trial=None, uid=None):
        """See :func:`~orion.storage.BaseStorageProtocol.get_trials`"""
        if trial is not None and uid is not None:
            assert trial.id == uid

        if uid is None:
            if trial is None:
                raise MissingArguments('trial or uid argument should be populated')

            uid = trial.id

        _hash, _rev = 0, 0
        data = uid.split('_', maxsplit=1)

        if len(data) == 1:
            _hash = data[0]

        elif len(data) == 2:
            _hash, _rev = data

        trials = self.backend.get_trial(TrackTrial(_hash=_hash, revision=_rev))

        if trials is None:
            return None

        assert len(trials) == 1
        return TrialAdapter(trials[0], objective=self.objective)

    def reserve_trial(self, experiment):
        """Select a pending trial and reserve it for the worker"""
        query = dict(
            group_id=experiment.id,
            status={'$in': ['new', 'suspended', 'interrupted']}
        )

        trial = self.backend.fetch_and_update_trial(
            query,
            'set_trial_status',
            status=get_track_status('reserved'))

        if trial is None:
            return None

        return TrialAdapter(trial, objective=self.objective)

    def fetch_lost_trials(self, experiment):
        """Fetch all trials that have a heartbeat older than
        some given time delta (2 minutes by default)
        """
        # TODO: Configure this
        threshold = to_epoch(datetime.datetime.utcnow() - datetime.timedelta(seconds=60 * 2))
        lte_comparison = {'$lte': threshold}
        query = {
            'experiment': experiment.id,
            'status': 'reserved',
            'heartbeat': lte_comparison
        }

        return self._fetch_trials(query)

    def push_trial_results(self, trial):
        """Push the trial's results to the database"""
        self.backend.log_trial_metrics()

    def fetch_noncompleted_trials(self, experiment):
        """Fetch all non completed trials"""
        query = dict(
            group_id=experiment.id,
            status={'$ne': get_track_status('completed')}
        )
        return self.backend.fetch_trials(query)

    def fetch_trial_by_status(self, experiment, status):
        """Fetch all trials with the given status"""
        trials = self._fetch_trials(dict(status=status, group_id=experiment.id))
        return trials

    def count_completed_trials(self, experiment):
        """Count the number of completed trials"""
        return len(self._fetch_trials(dict(status='completed', group_id=experiment.id)))

    def count_broken_trials(self, experiment):
        """Count the number of broken trials"""
        return len(self._fetch_trials(dict(status='broken', group_id=experiment.id)))

    def update_heartbeat(self, trial):
        """Update trial's heartbeat"""
        self.backend.log_trial_metadata(trial.storage,
                                        heartbeat=to_epoch(datetime.datetime.utcnow()))
