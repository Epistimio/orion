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
import math
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
                 'status', 'algorithms', '_db', '_init_done', '_id')

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
            config = sorted(config, key=lambda x: x['metadata']['datetime'],
                            reverse=True)[0]
            #  assert (len(config) == 1) is not True  # currently
            for attrname in self.__slots__:
                if not attrname.startswith('_'):
                    setattr(self, attrname, config[attrname])
            self._id = config['_id']

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
            exp_name=self.name,
            user=self.metadata['user'],
            status='new'
            )
        new_trials = Trial.build(self._db.read('trials', query))

        if not new_trials:
            return None

        if score_handle is None:
            selected_trial = random.sample(new_trials, 1)[0]
        else:
            raise NotImplementedError("scoring will be supported in the next iteration.")

        selected_trial.status = 'reserved'

        self._db.write('trials', dict(selected_trial), query={'_id': selected_trial.id})

        return selected_trial

    def push_completed_trial(self, trial):
        """Inform database about an evaluated `trial` with results.

        :param trial: Corresponds to a successful evaluation of a particular run.
        :type trial: `Trial`

        .. note:: Change status from *reserved* to *completed*.
        """
        trial.status = 'completed'
        self._db.write('trials', dict(trial), query={'_id': trial.id})

    def register_trials(self, trials):
        """Inform database about *new* suggested trial with specific parameter
        values. Each of them correspond to a different possible run.

        :type trials: list of `Trial`
        """
        for trial in trials:
            trial.status = 'new'
        self._db.write('trials', list(map(dict, trials)))

    @property
    def is_done(self):
        """Count how many trials have been completed and compare with `max_trials`.

        .. note:: To be used as a terminating condition in a ``Worker``.
        """
        query = dict(
            exp_name=self.name,
            user=self.metadata['user'],
            status='completed'
            )
        num_completed_trials = len(self._db.read('trials', query, {'_id': 1}))
        if num_completed_trials >= self.max_trials:
            return True
        return False

    @property
    def configuration(self):
        """Return a copy of an `Experiment` configuration as a dictionary."""
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
            if section == 'status' or \
                    section not in self.__slots__ or \
                    section.startswith('_'):
                continue
            setattr(self, section, value)

        self.status = 'new'
        final_config = self.configuration  # grab dict representation of Experiment

        # Sanitize and replace some sections with objects
        self._sanitize_config()

        # If everything is alright, push new config to database
        if is_new:
            self._db.write('experiments', final_config)
            # XXX: This may be MongoDB only; it updates an inserted dict with _id
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
            exp_name=self.name,
            user=self.metadata['user'],
            status='completed'
            )
        completed_trials = self._db.read('trials', query,
                                         selection={'_id': 1, 'end_time': 1,
                                                    'results': 1})
        stats = dict()
        stats['trials_completed'] = len(completed_trials)
        stats['best_trials_id'] = None
        stats['best_evaluation'] = math.inf
        stats['start_time'] = self.metadata['datetime']
        stats['finish_time'] = stats['start_time']
        for trial in completed_trials:
            if trial.end_time > stats['finish_time']:
                stats['finish_time'] = trial.end_time
            assert trial.results[0].type == 'objective'
            objective = trial.results[0].value
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
            # 'status' should not be in config
            # 'max_trials' overwrites without forking
            if section in ('status', 'max_trials') or \
                    section not in self.__slots__ or \
                    section.startswith('_'):
                continue
            if getattr(self, section) != value:
                is_diff = True
                break

        return is_diff
