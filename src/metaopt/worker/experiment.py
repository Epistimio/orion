# -*- coding: utf-8 -*-
"""
:mod:`metaopt.worker.experiment` -- Description of an optimization attempt
==========================================================================

.. module:: trial
   :platform: Unix
   :synopsis: Manage history of trials corresponding to a black box process

"""

import copy
import datetime
import getpass
import logging
import random

import six

from metaopt.io.database import Database
from metaopt.worker.trial import Trial

log = logging.getLogger(__name__)


class Experiment(object):
    """Represents an entry in database/experiments collection.

    Attributes
    ----------
    name : str
       Unique identifier for this experiment per `user`.
    refers : dict or list of `Experiment` objects, after initialization is done.
       A dictionary pointing to a past `Experiment` name, ``refers[name]``, whose
       trials we want to add in the history of completed trials we want to re-use.
    metadata : dict
       Contains managerial information about this `Experiment`.
    pool_size : int
       How many workers can participate asynchronously in this `Experiment`.
    max_trials : int
       How many trials must be evaluated, before considering this `Experiment` done.

       This attribute can be updated if the rest of the experiment configuration
       is the same. In that case, if trying to set to an already set experiment,
       it will overwrite the previous one.
    status : str
       A keyword among {*'pending'*, *'done'*, *'broken'*} indicating
       how **MetaOpt** considers the current `Experiment`. This attribute cannot
       be set from an mopt configuration.

       * 'pending' : Denotes an experiment with valid configuration which is
          currently being handled by **MetaOpt**.
       * 'done' : Denotes an experiment which has completed `max_trials` number
          of parameter evaluations and is not *pending*.
       * 'broken' : Denotes an experiment which stopped unsuccessfully due to
          unexpected behaviour.
    algorithms : dict of dicts or list of `Algorithm` objects, after initialization is done.
       Complete specification of the optimization and dynamical procedures taking
       place in this `Experiment`.

    Metadata
    --------
    user : str
       System user currently owning this running process, the one who invoked **MetaOpt**.
    datetime : `datetime.datetime`
       When was this particular configuration submitted to the database.
    mopt_version : str
       Version of **MetaOpt** which suggested this experiment. `user`'s current
       **MetaOpt** version.
    user_script : str
       Full absolute path to `user`'s executable.
    user_config : str
       Full absolute path to `user`'s configuration, possibly templated for **MetaOpt**.
    user_args : list of str
       Contains separate arguments to be passed when invoking `user_script`,
       possibly templated for **MetaOpt**.
    user_vcs : str, optional
       User's version control system for this executable's code repository.
    user_version : str, optional
       Current user's repository version.
    user_commit_hash : str, optional
       Current `Experiment`'s commit hash for **MetaOpt**'s invocation.

    """

    __slots__ = ('name', 'refers', 'metadata', 'pool_size', 'max_trials',
                 'status', 'algorithms', '_db', '_init_done', '_id', '_last_fetched')
    # 'status' should not be in config
    non_forking_attrs = ('status', 'pool_size', 'max_trials')

    def __init__(self, name):
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
        self._init_done = False
        self._db = Database()  # fetch database instance

        self._id = None
        self.name = name
        self.refers = None
        user = getpass.getuser()
        stamp = datetime.datetime.utcnow()
        self.metadata = {'user': user, 'datetime': stamp}
        self.pool_size = None
        self.max_trials = None
        self.status = None
        self.algorithms = None

        config = self._db.read('experiments',
                               {'name': name, 'metadata.user': user})
        if config:
            if len(config) > 1:
                log.warning("Many (%s) experiments for (%s, %s) are available but "
                            "only the most recent one can be accessed. "
                            "Experiment forks will be supported soon.", len(config), name, user)
            config = sorted(config, key=lambda x: x['metadata']['datetime'],
                            reverse=True)[0]
            for attrname in self.__slots__:
                if not attrname.startswith('_'):
                    setattr(self, attrname, config[attrname])
            self._id = config['_id']

        self._last_fetched = self.metadata['datetime']

    def reserve_trial(self, score_handle=None):
        """Find *new* trials that exist currently in database and select one of
        them based on the highest score return from `score_handle` callable.

        :param score_handle: A way to decide which trial out of the *new* ones to
           to pick as *reserved*, defaults to a random choice.
        :type score_handle: callable

        :return: selected `Trial` object, None if could not find any.
        """
        if score_handle is not None and not callable(score_handle):
            raise ValueError("Argument `score_handle` must be callable with a `Trial`.")

        query = dict(
            experiment=self._id,
            status={'$in': ['new', 'suspended', 'interrupted']}
            )
        new_trials = Trial.build(self._db.read('trials', query))

        if not new_trials:
            return None

        if score_handle is None:
            selected_trial = random.sample(new_trials, 1)[0]
        else:
            raise NotImplementedError("scoring will be supported in the next iteration.")

        if selected_trial.status == 'new':
            selected_trial.start_time = datetime.datetime.utcnow()
        selected_trial.status = 'reserved'

        self._db.write('trials', selected_trial.to_dict(),
                       query={'_id': selected_trial.id})

        return selected_trial

    def push_completed_trial(self, trial):
        """Inform database about an evaluated `trial` with results.

        :param trial: Corresponds to a successful evaluation of a particular run.
        :type trial: `Trial`

        .. note:: Change status from *reserved* to *completed*.
        """
        trial.end_time = datetime.datetime.utcnow()
        trial.status = 'completed'
        self._db.write('trials', trial.to_dict(), query={'_id': trial.id})

    def register_trials(self, trials):
        """Inform database about *new* suggested trial with specific parameter
        values. Each of them correspond to a different possible run.

        :type trials: list of `Trial`
        """
        stamp = datetime.datetime.utcnow()
        for trial in trials:
            trial.experiment = self._id
            trial.status = 'new'
            trial.submit_time = stamp
        trials_dicts = list(map(lambda x: x.to_dict(), trials))
        self._db.write('trials', trials_dicts)

    def fetch_completed_trials(self):
        """Fetch recent completed trials that this `Experiment` instance has not
        yet seen.

        .. note:: It will return only those with `Trial.end_time` after
           `_last_fetched`, for performance reasons.

        :return: list of completed `Trial` objects
        """
        query = dict(
            experiment=self._id,
            status='completed',
            end_time={'$gte': self._last_fetched}
            )
        completed_trials = Trial.build(self._db.read('trials', query))
        self._last_fetched = datetime.datetime.utcnow()

        return completed_trials

    @property
    def is_done(self):
        """Count how many trials have been completed and compare with `max_trials`.

        .. note:: To be used as a terminating condition in a ``Worker``.
        """
        query = dict(
            experiment=self._id,
            status='completed'
            )
        num_completed_trials = self._db.count('trials', query)
        if num_completed_trials >= self.max_trials:
            return True
        return False

    @property
    def configuration(self):
        """Return a copy of an `Experiment` configuration as a dictionary."""
        # After successful initialization, get a dict form from DB.
        # Check :attr:`algorithms` and :attr:`refers` to see why.
        if self._init_done:
            return self._db.read('experiments', {'_id': self._id})[0]

        config = dict()
        for attrname in self.__slots__:
            if not attrname.startswith('_'):
                config[attrname] = getattr(self, attrname)
        # Reason for deepcopy is that some attributes are dictionaries
        # themselves, we don't want to accidentally change the state of this
        # object from a getter.
        return copy.deepcopy(config)

    def configure(self, config):
        """Set `Experiment` by overwriting current attributes.

        If `Experiment` was already set and an overwrite is needed, a *fork*
        is advised with a different :attr:`name` for this particular configuration.

        .. note:: Calling this property is necessary for an experiment's
           initialization process to be considered as done. But it can be called
           only once.
        """
        if self._init_done:
            raise RuntimeError("Configuration is done; cannot reset an Experiment.")

        # If status is None in this object, then database did not hit a config
        # with same (name, user's name) pair. Everything depends on the user's
        # moptconfig to set.
        if self.status is None:
            if config['name'] != self.name or \
                    config['metadata']['user'] != self.metadata['user'] or \
                    config['metadata']['datetime'] != self.metadata['datetime']:
                raise ValueError("Configuration given is inconsistent with this Experiment.")
            is_new = True
        else:
            # Fork if it is needed
            is_new = self._is_different_from(config)
            if is_new:
                self._fork_config(config)  # Change (?) `name` attribute here.

        # Just overwrite everything given
        for section, value in six.iteritems(config):
            if section == 'status':
                log.warning("Found section 'status' in configuration. This experiment "
                            "attribute cannot be set. Ignoring.")
                continue
            if section not in self.__slots__:
                log.warning("Found section '%s' in configuration. Experiments "
                            "do not support this option. Ignoring.", section)
                continue
            if section.startswith('_'):
                log.warning("Found section '%s' in configuration. "
                            "Cannot set private attributes. Ignoring.", section)
                continue
            setattr(self, section, value)

        self.status = 'pending'
        final_config = self.configuration  # grab dict representation of Experiment

        # Sanitize and replace some sections with objects
        self._sanitize_config()

        # If everything is alright, push new config to database
        if is_new:
            self._db.write('experiments', final_config)
            # XXX: Reminder for future DB implementations:
            # MongoDB, updates an inserted dict with _id, so should you :P
            self._id = final_config['_id']
        else:
            self._db.write('experiments', final_config, {'_id': self._id})

        self._init_done = True

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
        query = dict(
            experiment=self._id,
            status='completed'
            )
        completed_trials = self._db.read('trials', query,
                                         selection={'_id': 1, 'end_time': 1,
                                                    'results': 1})
        stats = dict()
        stats['trials_completed'] = len(completed_trials)
        stats['best_trials_id'] = None
        trial = Trial(**completed_trials[0])
        stats['best_evaluation'] = trial.objective.value
        stats['best_trials_id'] = trial.id
        stats['start_time'] = self.metadata['datetime']
        stats['finish_time'] = stats['start_time']
        for trial in completed_trials:
            trial = Trial(**trial)
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

    def _sanitize_config(self):
        """Check before dispatching experiment whether configuration corresponds
        to a runnable experiment.

        1. Check `refers` and instantiate `Experiment` objects from it.
        2. From `metadata` given: ``user_script``, ``user_config`` should exist.
        3. Check if experiment `is_done`, prompt for larger `max_trials` if it is.
        4. Check whether configured algorithms correspond to [known]/valid
           implementations of the ``Algorithm`` class. Instantiate these objects.
        """
        pass

    def _fork_config(self, config):
        """Ask for a different identifier for this experiment. Set :attr:`refers`
        key to previous experiment's name, the one that we forked from.

        :param config: Conflicting configuration that will change based on prompt.
        """
        raise NotImplementedError()

    def _is_different_from(self, config):
        """Return True, if current `Experiment`'s configuration as described by
        its attributes is different from the one suggested in `config`.
        """
        is_diff = False
        for section, value in six.iteritems(config):
            if section in self.non_forking_attrs or \
                    section not in self.__slots__ or \
                    section.startswith('_'):
                continue
            if getattr(self, section) != value:
                is_diff = True
                break

        return is_diff
