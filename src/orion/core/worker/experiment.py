# -*- coding: utf-8 -*-
# pylint:disable=protected-access,too-many-public-methods,too-many-lines
"""
:mod:`orion.core.worker.experiment` -- Description of an optimization attempt
=============================================================================

.. module:: experiment
   :platform: Unix
   :synopsis: Manage history of trials corresponding to a black box process

"""
import copy
import datetime
import logging

import orion.core
from orion.core.evc.adapters import BaseAdapter
from orion.core.evc.experiment import ExperimentNode
from orion.storage.base import FailedUpdate, get_storage, ReadOnlyStorageProtocol

log = logging.getLogger(__name__)


# pylint: disable=too-many-public-methods
class Experiment:
    """Represents an entry in database/experiments collection.

    Attributes
    ----------
    name : str
       Unique identifier for this experiment per `user`.
    id: object
       id of the experiment in the database if experiment is configured. Value is `None`
       if the experiment is not configured.
    refers : dict or list of `Experiment` objects, after initialization is done.
       A dictionary pointing to a past `Experiment` id, ``refers[parent_id]``, whose
       trials we want to add in the history of completed trials we want to re-use.
       For convenience and database effiency purpose, all experiments of a common tree shares
       `refers[root_id]`, with the root experiment refering to itself.
    version: int
        Current version of this experiment.
    metadata : dict
       Contains managerial information about this `Experiment`.
    pool_size : int
       How many workers can participate asynchronously in this `Experiment`.
    max_trials : int
       How many trials must be evaluated, before considering this `Experiment` done.
       This attribute can be updated if the rest of the experiment configuration
       is the same. In that case, if trying to set to an already set experiment,
       it will overwrite the previous one.
    algorithms : dict of dicts or an `PrimaryAlgo` object, after initialization is done.
       Complete specification of the optimization and dynamical procedures taking
       place in this `Experiment`.

    Metadata
    --------
    user : str
       System user currently owning this running process, the one who invoked **Oríon**.
    datetime : `datetime.datetime`
       When was this particular configuration submitted to the database.
    orion_version : str
       Version of **Oríon** which suggested this experiment. `user`'s current
       **Oríon** version.
    user_script : str
       Full absolute path to `user`'s executable.
    user_args : list of str
       Contains separate arguments to be passed when invoking `user_script`,
       possibly templated for **Oríon**.
    user_vcs : str, optional
       User's version control system for this executable's code repository.
    user_version : str, optional
       Current user's repository version.
    user_commit_hash : str, optional
       Current `Experiment`'s commit hash for **Oríon**'s invocation.

    """

    __slots__ = ('name', 'refers', 'metadata', 'pool_size', 'max_trials', 'version',
                 'algorithms', 'producer', 'working_dir', '_init_done', '_id',
                 '_node', '_storage')
    non_branching_attrs = ('pool_size', 'max_trials')

    def __init__(self, name, version=None):
        self._init_done = True

        self._id = None
        self.name = name
        self.version = version if version else 1
        self._node = None
        self.refers = {}
        self.metadata = {}
        self.pool_size = None
        self.max_trials = None
        self.algorithms = None
        self.working_dir = None
        self.producer = {}
        # this needs to be an attribute because we override it in ExperienceView
        self._storage = get_storage()

        self._node = ExperimentNode(self.name, self.version, experiment=self)

    def fetch_trials(self, with_evc_tree=False):
        """Fetch all trials of the experiment"""
        return self._select_evc_call(with_evc_tree, 'fetch_trials')

    def get_trial(self, trial=None, uid=None):
        """Fetch a single Trial, see `orion.storage.base.BaseStorage.get_trial`"""
        return self._storage.get_trial(trial, uid)

    def retrieve_result(self, trial, *args, **kwargs):
        """See :func:`~orion.storage.BaseStorageProtocol.retrieve_result`"""
        return self._storage.retrieve_result(trial, *args, **kwargs)

    def set_trial_status(self, *args, **kwargs):
        """See :func:`~orion.storage.BaseStorageProtocol.set_trial_status`"""
        return self._storage.set_trial_status(*args, **kwargs)

    def reserve_trial(self, score_handle=None):
        """Find *new* trials that exist currently in database and select one of
        them based on the highest score return from `score_handle` callable.

        Parameters
        ----------
        score_handle: callable object, optional
            A way to decide which trial out of the *new* ones to
            to pick as *reserved*, defaults to a random choice.
            Deprecated

        Returns
        -------
        Selected `Trial` object, None if could not find any.

        """
        log.debug('reserving trial with (score: %s)', score_handle)

        if score_handle is not None:
            log.warning("Argument `score_handle` is deprecated")

        self.fix_lost_trials()

        selected_trial = self._storage.reserve_trial(self)
        log.debug('reserved trial (trial: %s)', selected_trial)
        return selected_trial

    def fix_lost_trials(self):
        """Find lost trials and set them to interrupted.

        A lost trial is defined as a trial whose heartbeat as not been updated since two times
        the wait time for monitoring. This usually means that the trial is stalling or has been
        interrupted in some way without its status being changed. This functions finds such
        trials and set them as interrupted so they can be launched again.

        """
        trials = self._storage.fetch_lost_trials(self)

        for trial in trials:
            log.debug('Setting lost trial %s status to interrupted...', trial.id)

            try:
                self._storage.set_trial_status(trial, status='interrupted')
                log.debug('success')
            except FailedUpdate:
                log.debug('failed')

    def update_completed_trial(self, trial, results_file):
        """Inform database about an evaluated `trial` with results.

        :param trial: Corresponds to a successful evaluation of a particular run.
        :type trial: `Trial`

        .. note::

            Change status from *reserved* to *completed*.

        """
        trial.status = 'completed'
        trial.end_time = datetime.datetime.utcnow()
        self._storage.retrieve_result(trial, results_file)
        # push trial results updates the entire trial status included
        self._storage.push_trial_results(trial)

    def register_lie(self, lying_trial):
        """Register a *fake* trial created by the strategist.

        The main difference between fake trial and orignal ones is the addition of a fake objective
        result, and status being set to completed. The id of the fake trial is different than the id
        of the original trial, but the original id can be computed using the hashcode on parameters
        of the fake trial. See mod:`orion.core.worker.strategy` for more information and the
        Strategist object and generation of fake trials.

        Parameters
        ----------
        trials: `Trial` object
            Fake trial to register in the database

        Raises
        ------
        orion.core.io.database.DuplicateKeyError
            If a trial with the same id already exist in the database. Since the id is computed
            based on a hashing of the trial, this should mean that an identical trial already exist
            in the database.

        """
        lying_trial.status = 'completed'
        lying_trial.end_time = datetime.datetime.utcnow()
        self._storage.register_lie(lying_trial)

    def register_trial(self, trial):
        """Register new trial in the database.

        Inform database about *new* suggested trial with specific parameter values. Trials may only
        be registered one at a time to avoid registration of duplicates.

        Parameters
        ----------
        trials: `Trial` object
            Trial to register in the database

        Raises
        ------
        orion.core.io.database.DuplicateKeyError
            If a trial with the same id already exist in the database. Since the id is computed
            based on a hashing of the trial, this should mean that an identical trial already exist
            in the database.

        """
        stamp = datetime.datetime.utcnow()
        trial.experiment = self._id
        trial.status = 'new'
        trial.submit_time = stamp

        self._storage.register_trial(trial)

    def _select_evc_call(self, with_evc_tree, function, *args, **kwargs):
        if self._node is not None and with_evc_tree:
            return getattr(self._node, function)(*args, **kwargs)

        return getattr(self._storage, function)(self, *args, **kwargs)

    def fetch_trials_by_status(self, status, with_evc_tree=False):
        """Fetch all trials with the given status

        Trials are sorted based on `Trial.submit_time`

        :return: list of `Trial` objects
        """
        return self._select_evc_call(with_evc_tree, 'fetch_trial_by_status', status)

    def fetch_noncompleted_trials(self, with_evc_tree=False):
        """Fetch non-completed trials of this `Experiment` instance.

        Trials are sorted based on `Trial.submit_time`

        .. note::

            It will return all non-completed trials, including new, reserved, suspended,
            interrupted and broken ones.

        :return: list of non-completed `Trial` objects
        """
        return self._select_evc_call(with_evc_tree, 'fetch_noncompleted_trials')

    # pylint: disable=invalid-name
    @property
    def id(self):
        """Id of the experiment in the database if configured.

        Value is `None` if the experiment is not configured.
        """
        return self._id

    @property
    def node(self):
        """Node of the experiment in the version control tree.

        Value is `None` if the experiment is not connected to the version control tree.

        .. seealso::

            :py:meth:`orion.core.worker.experiment.Experiment.connect_to_version_control_tree`

        """
        return self._node

    @property
    def is_done(self):
        """Return True, if this experiment is considered to be finished.

        1. Count how many trials have been completed and compare with `max_trials`.
        2. Ask `algorithms` if they consider there is a chance for further improvement.

        .. note::

            To be used as a terminating condition in a ``Worker``.

        """
        num_completed_trials = self._storage.count_completed_trials(self)

        return ((num_completed_trials >= self.max_trials) or
                (self._init_done and self.algorithms.is_done))

    @property
    def is_broken(self):
        """Return True, if this experiment is considered to be broken.

        Count how many trials are broken and return True if that number has reached
        as given threshold.


        """
        num_broken_trials = self._storage.count_broken_trials(self)
        return num_broken_trials >= orion.core.config.worker.max_broken

    @property
    def space(self):
        """Return problem's parameter `orion.algo.space.Space`.

        .. note:: It will return None, if experiment init is not done.
        """
        if self._init_done:
            return self.algorithms.space
        return None

    @property
    def configuration(self):
        """Return a copy of an `Experiment` configuration as a dictionary."""
        config = dict()
        for attrname in self.__slots__:
            if attrname.startswith('_'):
                continue
            attribute = copy.deepcopy(getattr(self, attrname))
            config[attrname] = attribute
            if self._init_done and attrname == 'algorithms':
                config[attrname] = attribute.configuration
            elif attrname == "refers" and isinstance(attribute.get("adapter"), BaseAdapter):
                config[attrname]['adapter'] = config[attrname]['adapter'].configuration
            elif self._init_done and attrname == "producer" and attribute.get("strategy"):
                config[attrname]['strategy'] = config[attrname]['strategy'].configuration

        if self.id:
            config['_id'] = self.id

        return copy.deepcopy(config)

    @property
    def stats(self):
        """Calculate a stats dictionary for this particular experiment.

        Returns
        -------
        stats : dict

        Stats
        -----
        trials_completed : int
           Number of completed trials
        best_trials_id : int
           Unique identifier of the `Trial` object in the database which achieved
           the best known objective result.
        best_evaluation : float
           Evaluation score of the best trial
        start_time : `datetime.datetime`
           When Experiment was first dispatched and started running.
        finish_time : `datetime.datetime`
           When Experiment reached terminating condition and stopped running.
        duration : `datetime.timedelta`
           Elapsed time.

        """
        completed_trials = self.fetch_trials_by_status('completed')

        if not completed_trials:
            return dict()
        stats = dict()
        stats['trials_completed'] = len(completed_trials)
        stats['best_trials_id'] = None
        trial = completed_trials[0]
        stats['best_evaluation'] = trial.objective.value
        stats['best_trials_id'] = trial.id
        stats['start_time'] = self.metadata['datetime']
        stats['finish_time'] = stats['start_time']
        for trial in completed_trials:
            # All trials are going to finish certainly after the start date
            # of the experiment they belong to
            if trial.end_time > stats['finish_time']:  # pylint:disable=no-member
                stats['finish_time'] = trial.end_time
            objective = trial.objective.value
            if objective < stats['best_evaluation']:
                stats['best_evaluation'] = objective
                stats['best_trials_id'] = trial.id
        stats['duration'] = stats['finish_time'] - stats['start_time']

        return stats

    def __repr__(self):
        """Represent the object as a string."""
        return "Experiment(name=%s, metadata.user=%s, version=%s)" % \
            (self.name, self.metadata['user'], self.version)


# pylint: disable=too-few-public-methods
class ExperimentView(object):
    """Non-writable view of an experiment

    .. seealso::

        :py:class:`orion.core.worker.experiment.Experiment` for writable experiments.

    """

    __slots__ = ('_experiment', )

    #                     Attributes
    valid_attributes = (["_id", "name", "refers", "metadata", "pool_size", "max_trials",
                         "version", "space"] +
                        # Properties
                        ["id", "node", "is_done", "space", "algorithms", "stats", "configuration"] +
                        # Methods
                        ["fetch_trials", "fetch_trials_by_status", "get_trial"])

    def __init__(self, experiment):
        """TODO"""
        self._experiment = experiment
        self._experiment._storage = ReadOnlyStorageProtocol(experiment._storage)

    def __getattr__(self, name):
        """Get attribute only if valid"""
        if name not in self.valid_attributes:
            raise AttributeError("Cannot access attribute %s on view-only experiments." % name)

        return getattr(self._experiment, name)

    def __repr__(self):
        """Represent the object as a string."""
        return "ExperimentView(name=%s, metadata.user=%s, version=%s)" % \
            (self.name, self.metadata['user'], self.version)
