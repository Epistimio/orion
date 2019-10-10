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
import getpass
import logging
import sys

import orion.core
from orion.core.cli.evc import fetch_branching_configuration
from orion.core.evc.adapters import Adapter, BaseAdapter
from orion.core.evc.conflicts import detect_conflicts, ExperimentNameConflict
from orion.core.io.database import DuplicateKeyError
from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder
from orion.core.io.interactive_commands.branching_prompt import BranchingPrompt
from orion.core.io.space_builder import SpaceBuilder
import orion.core.utils.backward as backward
from orion.core.utils.exceptions import RaceCondition
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.core.worker.strategy import (BaseParallelStrategy,
                                        Strategy)
from orion.storage.base import get_storage, ReadOnlyStorageProtocol

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

    def __init__(self, name, user=None, version=None):
        """Initialize an Experiment object with primary key (:attr:`name`, :attr:`user`).

        Try to find an entry in `Database` with such a key and config this object
        from it import, if successful. Else, init with default/empty values and
        insert new entry with this object's attributes in database.

        .. note::
           Practically initialization has not finished until `config`'s setter
           is called.

        :param name: Describe a configuration with a unique identifier per :attr:`user`.
        :type name: str
        """
        log.debug("Creating Experiment object with name: %s", name)
        self._init_done = False

        self._id = None
        self.name = name
        self._node = None
        self.refers = {}
        if user is None:
            user = getpass.getuser()
        self.metadata = {'user': user}
        self.pool_size = None
        self.max_trials = None
        self.algorithms = None
        self.working_dir = None
        self.producer = {'strategy': None}
        self.version = 1
        # this needs to be an attribute because we override it in ExperienceView
        self._storage = get_storage()

        config = self._storage.fetch_experiments({'name': name})

        if config:
            log.debug("Found existing experiment, %s, under user, %s, registered in database.",
                      name, user)

            if len(config) > 1:
                max_version = max(config, key=lambda exp: exp.get('version', 1)).get('version', 1)

                if version is None:
                    self.version = max_version
                else:
                    self.version = version

                if self.version > max_version:
                    log.warning("Version %s was specified but most recent version is only %s. "
                                "Using %s.", self.version, max_version, max_version)

                self.version = min(self.version, max_version)

                log.info("Many versions for experiment %s have been found. Using latest "
                         "version %s.", name, self.version)
                config = filter(lambda exp: exp.get('version', 1) == self.version, config)

            config = sorted(config, key=lambda x: x['metadata']['datetime'],
                            reverse=True)[0]

            backward.populate_priors(config['metadata'])

            for attrname in self.__slots__:
                if not attrname.startswith('_') and attrname in config:
                    setattr(self, attrname, config[attrname])
            self._id = config['_id']

    def fetch_trials(self, with_evc_tree=False):
        """Fetch all trials of the experiment"""
        return self._select_evc_call(with_evc_tree, 'fetch_trials')

    def get_trial(self, trial=None, uid=None):
        """Fetch a single Trial, see `orion.storage.base.BaseStorage.get_trial`"""
        return self._storage.get_trial(trial, uid)

    def connect_to_version_control_tree(self, node):
        """Connect the experiment to its node in a version control tree

        .. seealso::

            :class:`orion.core.evc.experiment.ExperimentNode`

        :param node: Node giving access to the experiment version control tree.
        :type name: None or `ExperimentNode`
        """
        self._node = node

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

            updated = self._storage.set_trial_status(trial, status='interrupted')
            log.debug('success' if updated else 'failed')

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
            attribute = getattr(self, attrname)
            if self._init_done and attrname == 'algorithms':
                config[attrname] = attribute.configuration
            else:
                config[attrname] = attribute

            if attrname == "refers" and isinstance(attribute.get("adapter"), BaseAdapter):
                config[attrname] = copy.deepcopy(config[attrname])
                config[attrname]['adapter'] = config[attrname]['adapter'].configuration

            if self._init_done and attrname == "producer" and attribute.get("strategy"):
                config[attrname] = copy.deepcopy(config[attrname])
                config[attrname]['strategy'] = config[attrname]['strategy'].configuration

        # Reason for deepcopy is that some attributes are dictionaries
        # themselves, we don't want to accidentally change the state of this
        # object from a getter.
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

    def configure(self, config, enable_branching=True, enable_update=True):
        """Set `Experiment` by overwriting current attributes.

        If `Experiment` was already set and an overwrite is needed, a *branch*
        is advised with a different :attr:`name` for this particular configuration.

        .. note::

            Calling this property is necessary for an experiment's initialization process to be
            considered as done. But it can be called only once.

        """
        log.debug('configuring (name: %s)', config['name'])
        if self._init_done:
            raise RuntimeError("Configuration is done; cannot reset an Experiment.")

        # Experiment was build using db, but config was build before experiment got in db.
        # Fake a DuplicateKeyError to force reinstantiation of experiment with proper config.
        if self._id is not None and "datetime" not in config['metadata']:
            raise DuplicateKeyError("Cannot register an existing experiment with a new config")

        # Copy and simulate instantiating given configuration
        experiment = Experiment(self.name, version=self.version)
        experiment._instantiate_config(self.configuration)
        experiment._instantiate_config(config)
        experiment._init_done = True

        # If id is None in this object, then database did not hit a config
        # with same (name, user's name) pair. Everything depends on the user's
        # orion_config to set.
        if self._id is None:
            if config['name'] != self.name or \
                    config['metadata']['user'] != self.metadata['user']:
                raise ValueError("Configuration given is inconsistent with this Experiment.")
            must_branch = True
        else:
            # Branch if it is needed
            # TODO: When refactoring experiment managenent, is_different_from
            # will be used when EVC is not available.
            # must_branch = self._is_different_from(experiment.configuration)
            branching_configuration = fetch_branching_configuration(config)
            configuration = self.configuration
            configuration['_id'] = self._id
            conflicts = detect_conflicts(configuration, experiment.configuration)
            must_branch = len(conflicts.get()) > 1 or branching_configuration.get('branch')

            name_conflict = conflicts.get([ExperimentNameConflict])[0]
            if not name_conflict.is_resolved and not config.get('version'):
                raise RaceCondition('There was likely a race condition during version increment.')

            elif must_branch and not enable_branching:
                raise ValueError("Configuration is different and generate a branching event")

            elif must_branch:
                experiment._branch_config(conflicts, branching_configuration)

        final_config = experiment.configuration
        self._instantiate_config(final_config)

        self._init_done = True

        if not enable_update:
            return

        # If everything is alright, push new config to database
        if must_branch:
            final_config['metadata']['datetime'] = datetime.datetime.utcnow()
            self.metadata['datetime'] = final_config['metadata']['datetime']
            # This will raise DuplicateKeyError if a concurrent experiment with
            # identical (name, metadata.user) is written first in the database.
            self._storage.create_experiment(final_config)

            # XXX: Reminder for future DB implementations:
            # MongoDB, updates an inserted dict with _id, so should you :P
            self._id = final_config['_id']

            # Update refers in db if experiment is root
            if self.refers['parent_id'] is None:
                log.debug('update refers (name: %s)', config['name'])
                self.refers['root_id'] = self._id
                self._storage.update_experiment(self, refers=self.configuration['refers'])

        else:
            # Writing the final config to an already existing experiment raises
            # a DuplicatKeyError because of the embedding id `metadata.user`.
            # To avoid this `final_config["name"]` is popped out before
            # `db.write()`, thus seamingly breaking  the compound index
            # `(name, metadata.user)`
            log.debug('updating experiment (name: %s)', config['name'])

            final_config.pop("name")
            self._storage.update_experiment(self, **final_config)

    def _instantiate_config(self, config):
        """Check before dispatching experiment whether configuration corresponds
        to a executable experiment environment.

        1. Check `refers` and instantiate `Adapter` objects from it.
        2. Try to build parameter space from user arguments.
        3. Check whether configured algorithms correspond to [known]/valid
           implementations of the ``Algorithm`` class. Instantiate these objects.
        4. Check if experiment `is_done`, prompt for larger `max_trials` if it is. (TODO)

        """
        # Just overwrite everything else given
        for section, value in config.items():
            if section not in self.__slots__:
                log.info("Found section '%s' in configuration. Experiments "
                         "do not support this option. Ignoring.", section)
                continue
            if section.startswith('_'):
                log.info("Found section '%s' in configuration. "
                         "Cannot set private attributes. Ignoring.", section)
                continue

            # Copy sub configuration to value confusing side-effects
            # Only copy at this level, not `config` directly to avoid TypeErrors if config contains
            # non-serializable objects (copy.deepcopy complains otherwise).
            if isinstance(value, dict):
                value = copy.deepcopy(value)

            setattr(self, section, value)

        # TODO: Can we get rid of this try-except clause?
        try:
            space_builder = SpaceBuilder()
            space = space_builder.build(config['metadata']['priors'])

            if not space:
                raise ValueError("Parameter space is empty. There is nothing to optimize.")

            # Instantiate algorithms
            self.algorithms = PrimaryAlgo(space, self.algorithms)
        except KeyError:
            pass

        self.refers.setdefault('parent_id', None)
        self.refers.setdefault('root_id', self._id)
        self.refers.setdefault('adapter', [])
        if not isinstance(self.refers.get('adapter'), BaseAdapter):
            self.refers['adapter'] = Adapter.build(self.refers['adapter'])

        if not self.producer.get('strategy'):
            self.producer = {'strategy': Strategy(of_type="MaxParallelStrategy")}
        elif not isinstance(self.producer.get('strategy'), BaseParallelStrategy):
            self.producer = {'strategy': Strategy(of_type=self.producer['strategy'])}

    def _branch_config(self, conflicts, branching_configuration):
        """Ask for a different identifier for this experiment. Set :attr:`refers`
        key to previous experiment's name, the one that we branched from.

        :param config: Conflicting configuration that will change based on prompt.
        """
        experiment_brancher = ExperimentBranchBuilder(conflicts, branching_configuration)

        needs_manual_resolution = (not experiment_brancher.is_resolved or
                                   experiment_brancher.manual_resolution)

        if needs_manual_resolution:
            branching_prompt = BranchingPrompt(experiment_brancher)

            if not sys.__stdin__.isatty():
                raise ValueError(
                    "Configuration is different and generates a branching event:\n{}".format(
                        branching_prompt.get_status()))

            branching_prompt.cmdloop()

            if branching_prompt.abort or not experiment_brancher.is_resolved:
                sys.exit()

        adapter = experiment_brancher.create_adapters()
        self._instantiate_config(experiment_brancher.conflicting_config)
        self.refers['adapter'] = adapter
        self.refers['parent_id'] = self._id

    def _is_different_from(self, config):
        """Return True, if current `Experiment`'s configuration as described by
        its attributes is different from the one suggested in `config`.
        """
        is_diff = False
        for section, value in config.items():
            if section in self.non_branching_attrs or \
                    section not in self.__slots__ or \
                    section.startswith('_'):
                continue
            item = getattr(self, section)
            if item != value:
                log.warning("Config given is different from config found in db at section: %s",
                            section)
                log.warning("Config+ :\n%s", value)
                log.warning("Config- :\n%s", item)
                is_diff = True
                break

        return is_diff

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
                         "version"] +
                        # Properties
                        ["id", "node", "is_done", "space", "algorithms", "stats", "configuration"] +
                        # Methods
                        ["fetch_trials", "fetch_trials_by_status",
                         "connect_to_version_control_tree", "get_trial"])

    def __init__(self, name, user=None, version=None):
        """Initialize viewed experiment object with primary key (:attr:`name`, :attr:`user`).

        Build an experiment from configuration found in `Database` with a key (name, user).

        .. note::

            A view is fully configured at initialiation. It cannot be reconfigured.
            If no experiment is found for the key (name, user), a `ValueError` will be raised.

        :param name: Describe a configuration with a unique identifier per :attr:`user`.
        :type name: str
        """
        self._experiment = Experiment(name, user, version)

        if self._experiment.id is None:
            raise ValueError("No experiment with given name '%s' for user '%s' inside database, "
                             "no view can be created." %
                             (self._experiment.name, self._experiment.metadata['user']))

        # TODO: Views are not fully configured until configuration is refactored
        #       This snippet is to instantiate adapters anyhow, because it is required for
        #       experiment views in EVC.
        self.refers.setdefault('parent_id', None)
        self.refers.setdefault('root_id', self._id)
        self.refers.setdefault('adapter', [])
        if not isinstance(self.refers.get('adapter'), BaseAdapter):
            self.refers['adapter'] = Adapter.build(self.refers['adapter'])

        # try:
        #     self._experiment.configure(self._experiment.configuration, enable_branching=False,
        #                                enable_update=False)
        # except ValueError as e:
        #     if "Configuration is different and generates a branching event" in str(e):
        #         raise RuntimeError(
        #             "Configuration in the database does not correspond to the one generated by "
        #             "Experiment object. This is likely due to a backward incompatible update in "
        #             "Oríon. Please report to https://github.com/epistimio/orion/issues.") from e
        #     raise
        self._experiment._storage = ReadOnlyStorageProtocol(get_storage())

    def __getattr__(self, name):
        """Get attribute only if valid"""
        if name not in self.valid_attributes:
            raise AttributeError("Cannot access attribute %s on view-only experiments." % name)

        return getattr(self._experiment, name)

    def __repr__(self):
        """Represent the object as a string."""
        return "ExperimentView(name=%s, metadata.user=%s, version=%s)" % \
            (self.name, self.metadata['user'], self.version)
