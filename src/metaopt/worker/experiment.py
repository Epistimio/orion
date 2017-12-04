# -*- coding: utf-8 -*-
"""
:mod:`metaopt.worker.experiment` -- Description of an optimization attempt
==========================================================================

.. module:: trial
   :platform: Unix
   :synopsis: Manage history of trials corresponding to a black box process

"""

import datetime
import getpass

from metaopt.io.database import Database
#  from metaopt.worker.trial import Trial


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
       it will be the previous number of `max_trials` plus the one suggested by
       the input config.
    status : str
       A keyword among {*'new'*, *'running'*, *'done'*, *'broken'*} indicating
       how **MetaOpt** considers the current `Experiment`.

       * 'new' : Denotes a new valid configuration that has not been run yet.
       * 'running' : Denotes an experiment which is currently being executed.
       * 'done' : Denotes an experiment which has completed `max_trials` number
          of parameter evaluations and is not running.
       * 'broken' : Denotes an experiment with non valid configuration, which shall
          not be run.
    algorithms : dict of dicts or list of `Algorithm` objects, after initialization is done.
       Complete specification of the optimization and dynamical procedures taking
       place in this `Experiment`.

    Metadata
    --------
    user : str
       System user currently owning this running process, the one who invoked **MetaOpt**.
    datetime : str
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
            for varname in self.__slots__:
                if not varname.startswith('_'):
                    setattr(self, varname, config[varname])
            self._id = config['_id']

    def reserve_trial(self, score_handle=None):
        """Find *new* trials that exist currently in database and select one of
        them based on the highest score return from `score_handle` callable.

        :param score_handle: A way to decide which trial out of the *new* ones to
           to pick as *pending*, defaults to a random choice.
        :type score_handle: callable

        :return: selected `Trial` object
        """
        raise NotImplementedError()

    def push_completed_trial(self, trial):
        """Inform database about an evaluated `trial` with results.

        :param trial: Corresponds to a successful evaluation of a particular run.
        :type trial: `Trial`

        .. note:: Change status from *pending* to *completed*.
        """
        raise NotImplementedError()

    def register_trials(self, trials):
        """Inform database about *new* suggested trial with specific parameter
        values. Each of them correspond to a different possible run.

        :type trials: list of `Trial`
        """
        raise NotImplementedError()

    @property
    def is_done(self):
        """Count how many trials have been completed and compare with `max_trials`.

        .. note:: To be used as a terminating condition in a ``Worker``.
        """
        raise NotImplementedError()

    @property
    def config(self):
        """Return a dictionary no-writeable view of an `Experiment` configuration."""
        raise NotImplementedError()

    @config.setter
    def config(self, config):
        """Set `Experiment` by overwriting current attributes.

        If `Experiment` was already set and an overwrite is needed, a *fork*
        is advised with a different :attr:`name` for this particular configuration.

        .. note:: Calling this property is necessary for an experiment's
           initialization process to be considered as done.
        """
        raise NotImplementedError()

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
        start_time : `datetime.datetime`
           When Experiment was first dispatched and started running.
        finish_time : `datetime.datetime`
           When Experiment reached terminating condition and stopped running.
        duration : `datetime.timedelta`
           Elapsed time.

        """
        raise NotImplementedError()

    def _sanitize_config(self):
        """Chech before dispatching experiment whether configuration corresponds
        to a runnable experiment.

        For example, check whether configured algorithms correspond to [known]
        implementations of the ``Algorithm`` class. Instantiate these objects.
        """
        raise NotImplementedError()

    def _fork_config(self):
        """Ask for a different identifier for this experiment. Set :attr:`refers`
        key to previous experiment's name, the one that we forked from.
        """
        raise NotImplementedError()

    def _is_different_from(self, config):
        """Return True, if current `Experiment`'s configuration as described by
        its attributes is different from the one suggested in `config`.
        """
        raise NotImplementedError()
