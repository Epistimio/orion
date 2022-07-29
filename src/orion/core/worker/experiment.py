# pylint:disable=protected-access,too-many-public-methods,too-many-lines
"""
Description of an optimization attempt
======================================

Manage history of trials corresponding to a black box process.

"""
from __future__ import annotations

import contextlib
import copy
import datetime
import inspect
import logging
from dataclasses import dataclass, field

import pandas
from typing_extensions import Literal

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.evc.adapters import BaseAdapter
from orion.core.evc.experiment import ExperimentNode
from orion.core.io.database import DuplicateKeyError
from orion.core.utils.exceptions import UnsupportedOperation
from orion.core.utils.flatten import flatten
from orion.storage.base import BaseStorageProtocol, FailedUpdate

log = logging.getLogger(__name__)
Mode = Literal["r", "w", "x"]


@dataclass
class ExperimentStats:
    """
    Parameters
    ----------
    trials_completed: int
       Number of completed trials
    best_trials_id: int
       Unique identifier of the :class:`orion.core.worker.trial.Trial` object in the database
       which achieved the best known objective result.
    best_evaluation: float
       Evaluation score of the best trial
    start_time: `datetime.datetime`
       When Experiment was first dispatched and started running.
    finish_time: `datetime.datetime`
       When Experiment reached terminating condition and stopped running.
    duration: `datetime.timedelta`
       Elapsed time.
    """

    trials_completed: int
    best_trials_id: int
    best_evaluation: float
    start_time: datetime.datetime = field(default_factory=datetime.datetime)
    finish_time: datetime.datetime = field(default_factory=datetime.datetime)
    duration: datetime.timedelta = field(default_factory=datetime.timedelta)


# pylint: disable=too-many-public-methods
class Experiment:
    """Represents an entry in database/experiments collection.

    Attributes
    ----------
    name: str
       Unique identifier for this experiment per ``user``.
    id: object
       id of the experiment in the database if experiment is configured. Value is ``None``
       if the experiment is not configured.
    refers: dict or list of `Experiment` objects, after initialization is done.
       A dictionary pointing to a past `Experiment` id, ``refers[parent_id]``, whose
       trials we want to add in the history of completed trials we want to re-use.
       For the purpose of convenience and database efficiency, all experiments of a common tree
       share a ``refers[root_id]``, with the root experiment referring to itself.
    version: int
        Current version of this experiment.
    metadata: dict
       Contains managerial information about this `Experiment`.
    max_trials: int
       How many trials must be evaluated, before considering this `Experiment` done.
       This attribute can be updated if the rest of the experiment configuration
       is the same. In that case, if trying to set to an already set experiment,
       it will overwrite the previous one.
    max_broken: int
       How many trials must be broken, before considering this `Experiment` broken.
       This attribute can be updated if the rest of the experiment configuration
       is the same. In that case, if trying to set to an already set experiment,
       it will overwrite the previous one.
    space: Space
       Object representing the optimization space.
    algorithms: `BaseAlgorithm` object or a wrapper.
       Complete specification of the optimization and dynamical procedures taking
       place in this `Experiment`.

    Notes
    -----

    The following list represents possible entries in the metadata dict.

    user: str
       System user currently owning this running process, the one who invoked **Oríon**.
    datetime: `datetime.datetime`
       When was this particular configuration submitted to the database.
    orion_version: str
       Version of **Oríon** which suggested this experiment. `user`'s current
       **Oríon** version.
    user_script: str
       Full absolute path to `user`'s executable.
    user_args: list of str
       Contains separate arguments to be passed when invoking `user_script`,
       possibly templated for **Oríon**.
    user_vcs: str, optional
       User's version control system for this executable's code repository.
    user_version: str, optional
       Current user's repository version.
    user_commit_hash: str, optional
       Current `Experiment`'s commit hash for **Oríon**'s invocation.

    """

    __slots__ = (
        "name",
        "refers",
        "metadata",
        "max_trials",
        "max_broken",
        "version",
        "space",
        "algorithms",
        "working_dir",
        "_id",
        "_storage",
        "_node",
        "_mode",
    )
    non_branching_attrs = ("max_trials", "max_broken")

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        name: str,
        space: Space,
        version: int | None = 1,
        mode: Mode = "r",
        _id: str | int | None = None,
        max_trials: int | None = None,
        max_broken: int | None = None,
        algorithms: BaseAlgorithm | None = None,
        working_dir: str | None = None,
        metadata: dict | None = None,
        refers: dict | None = None,
        storage: BaseStorageProtocol | None = None,
    ):
        self._id = _id
        self.name = name
        self.space: Space = space
        self.version = version if version else 1
        self._mode = mode
        self.refers = refers or {}
        self.metadata = metadata or {}
        self.max_trials = max_trials
        self.max_broken = max_broken
        self.algorithms = algorithms
        self.working_dir = working_dir

        self._storage = storage

        self._node = ExperimentNode(
            self.name, self.version, experiment=self, storage=self._storage
        )

    @property
    def storage(self):
        """Return the storage currently in use by this experiment"""
        return self._storage

    def _check_if_writable(self):
        if self.mode == "r":
            calling_function = inspect.stack()[1].function
            raise UnsupportedOperation(
                f"Experiment must have write rights to execute `{calling_function}()`"
            )

    def _check_if_executable(self):
        if self.mode != "x":
            calling_function = inspect.stack()[1].function
            raise UnsupportedOperation(
                f"Experiment must have execution rights to execute `{calling_function}()`"
            )

    def __getstate__(self):
        """Remove storage instance during experiment serialization"""
        state = {}
        for entry in self.__slots__:
            state[entry] = getattr(self, entry)

        return state

    def __setstate__(self, state):
        for entry in self.__slots__:
            setattr(self, entry, state[entry])

    def to_pandas(self, with_evc_tree=False):
        """Builds a dataframe with the trials of the experiment

        Parameters
        ----------
        with_evc_tree: bool, optional
            Fetch all trials from the EVC tree.
            Default: False

        """
        columns = [
            "id",
            "experiment_id",
            "status",
            "suggested",
            "reserved",
            "completed",
            "objective",
        ]

        data = []
        for trial in self.fetch_trials(with_evc_tree=with_evc_tree):
            row = [
                trial.id,
                trial.experiment,
                trial.status,
                trial.submit_time,
                trial.start_time,
                trial.end_time,
            ]
            row.append(trial.objective.value if trial.objective else None)
            params = flatten(trial.params)
            for name in self.space.keys():
                row.append(params[name])

            data.append(row)

        columns += list(self.space.keys())

        if not data:
            return pandas.DataFrame([], columns=columns)

        return pandas.DataFrame(data, columns=columns)

    def fetch_trials(self, with_evc_tree=False):
        """Fetch all trials of the experiment"""
        return self._select_evc_call(with_evc_tree, "fetch_trials")

    def get_trial(self, trial=None, uid=None):
        """Fetch a single Trial, see :meth:`orion.storage.base.BaseStorageProtocol.get_trial`"""
        return self._storage.get_trial(trial, uid, experiment_uid=self.id)

    def retrieve_result(self, trial, *args, **kwargs):
        """See :meth:`orion.storage.base.BaseStorageProtocol.retrieve_result`"""
        return self._storage.retrieve_result(trial, *args, **kwargs)

    def set_trial_status(self, *args, **kwargs):
        """See :meth:`orion.storage.base.BaseStorageProtocol.set_trial_status`"""
        self._check_if_writable()
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
        self._check_if_executable()
        log.debug("reserving trial with (score: %s)", score_handle)

        self.fix_lost_trials()

        self.duplicate_pending_trials()

        selected_trial = self._storage.reserve_trial(self)
        log.debug("reserved trial (trial: %s)", selected_trial)
        return selected_trial

    def fix_lost_trials(self, with_evc_tree=True):
        """Find lost trials and set them to interrupted.

        A lost trial is defined as a trial whose heartbeat as not been updated since two times
        the wait time for monitoring. This usually means that the trial is stalling or has been
        interrupted in some way without its status being changed. This functions finds such
        trials and set them as interrupted so they can be launched again.

        """
        self._check_if_writable()

        if self._node is not None and with_evc_tree:
            for experiment in self._node.root:
                if experiment.item is self:
                    continue

                # Ugly hack to allow resetting parent's lost trials.
                experiment.item._mode = "w"
                experiment.item.fix_lost_trials(with_evc_tree=False)
                experiment.item._mode = "r"

        trials = self.fetch_lost_trials(with_evc_tree=False)

        for trial in trials:
            log.debug("Setting lost trial %s status to interrupted...", trial.id)

            try:
                self._storage.set_trial_status(trial, status="interrupted")
                log.debug("success")
            except FailedUpdate:
                log.debug("failed")

    def duplicate_pending_trials(self):
        """Find pending trials in EVC and duplicate them in current experiment.

        An experiment cannot execute trials from parent experiments otherwise some trials
        may have been executed in different environements of different experiment although they
        belong to the same experiment. Instead, trials that are pending in parent and child
        experiment are copied over to current experiment so that it can be reserved and executed.
        The parent or child experiment will only see their original copy of the trial, and
        the current experiment will only see the new copy of the trial.
        """
        self._check_if_writable()
        evc_pending_trials = self._select_evc_call(
            with_evc_tree=True, function="fetch_pending_trials"
        )
        exp_pending_trials = self._select_evc_call(
            with_evc_tree=False, function="fetch_pending_trials"
        )

        exp_trials_ids = {trial.id for trial in exp_pending_trials}

        for trial in evc_pending_trials:
            if trial.id in exp_trials_ids:
                continue

            trial.experiment = self.id
            trial.id_override = None
            # Danger danger, race conditions!
            try:
                self._storage.register_trial(trial)
            except DuplicateKeyError:
                log.debug("Race condition while trying to duplicate trial %s", trial.id)

    # pylint:disable=unused-argument
    def update_completed_trial(self, trial, results_file=None):
        """Inform database about an evaluated `trial` with results.

        :param trial: Corresponds to a successful evaluation of a particular run.
        :type trial: `Trial`

        .. note::

            Change status from *reserved* to *completed*.

        """
        self._check_if_executable()
        trial.status = "completed"
        trial.end_time = datetime.datetime.utcnow()
        self._storage.retrieve_result(trial)
        # push trial results updates the entire trial status included
        log.info("Completed trials with results: %s", trial.results)
        self._storage.push_trial_results(trial)

    def register_trial(self, trial, status="new"):
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
        self._check_if_writable()
        stamp = datetime.datetime.utcnow()
        trial.experiment = self._id
        trial.status = status
        trial.submit_time = stamp
        trial.exp_working_dir = self.working_dir

        self._storage.register_trial(trial)

    @contextlib.contextmanager
    def acquire_algorithm_lock(
        self, timeout: int | float = 60, retry_interval: int | float = 1
    ):
        """Acquire lock on algorithm

        This method should be called using a ``with``-clause.

        The context manager returns the algorithm object with its state updated
        based on the state loaded from storage.

        Upon leaving the context manager, the new state of the algorithm is saved back
        to the storage before releasing the lock.

        Parameters
        ----------
        timeout: int, optional
            Timeout for the acquisition of the lock. If the lock is not
            obtained before ``timeout``, then ``LockAcquisitionTimeout`` is raised.
            The timeout is only for the acquisition of the lock.
            Once the lock is obtained, it is valid until the context manager is closed.
            Default: 600.
        retry_interval: int, optional
            Sleep time between each attempts at acquiring the lock. Default: 1

        Raises
        ------
        ``RuntimeError``
            The algorithm configuration is different then the one during last execution of that
            same experiment.
        ``orion.storage.base.LockAcquisitionTimeout``
            The lock could not be obtained in less than ``timeout`` seconds.
        """

        self._check_if_writable()

        with self._storage.acquire_algorithm_lock(
            experiment=self, timeout=timeout, retry_interval=retry_interval
        ) as locked_algorithm_state:

            if locked_algorithm_state.configuration != self.algorithms.configuration:
                log.warning(
                    "Saved configuration: %s", locked_algorithm_state.configuration
                )
                log.warning(
                    "Current configuration: %s %s",
                    self.algorithms.configuration,
                    self._storage._db,
                )
                raise RuntimeError(
                    "Algorithm configuration changed since last experiment execution. "
                    "Algorithm cannot be resumed with a different configuration. "
                )

            if locked_algorithm_state.state:
                self.algorithms.set_state(locked_algorithm_state.state)

            yield self.algorithms

            locked_algorithm_state.set_state(self.algorithms.state_dict)

    def _select_evc_call(self, with_evc_tree, function, *args, **kwargs):
        if self._node is not None and with_evc_tree:
            return getattr(self._node, function)(*args, **kwargs)

        return getattr(self._storage, function)(self, *args, **kwargs)

    def fetch_trials_by_status(self, status, with_evc_tree=False):
        """Fetch all trials with the given status

        Trials are sorted based on `Trial.submit_time`

        :return: list of `Trial` objects
        """
        return self._select_evc_call(with_evc_tree, "fetch_trials_by_status", status)

    def fetch_pending_trials(self, with_evc_tree=False):
        """Fetch all trials with status new, interrupted or suspended

        Trials are sorted based on `Trial.submit_time`

        :return: list of `Trial` objects
        """
        return self._select_evc_call(with_evc_tree, "fetch_pending_trials")

    def fetch_lost_trials(self, with_evc_tree=False):
        """Fetch all reserved trials that are lost (old heartbeat)

        Trials are sorted based on `Trial.submit_time`

        :return: list of `Trial` objects
        """
        return self._select_evc_call(with_evc_tree, "fetch_lost_trials")

    def fetch_noncompleted_trials(self, with_evc_tree=False):
        """Fetch non-completed trials of this `Experiment` instance.

        Trials are sorted based on `Trial.submit_time`

        .. note::

            It will return all non-completed trials, including new, reserved, suspended,
            interrupted and broken ones.

        :return: list of non-completed `Trial` objects
        """
        return self._select_evc_call(with_evc_tree, "fetch_noncompleted_trials")

    # pylint: disable=invalid-name
    @property
    def id(self):
        """Id of the experiment in the database if configured.

        Value is `None` if the experiment is not configured.
        """
        return self._id

    @property
    def mode(self):
        """Return the access right of the experiment

        {'r': read, 'w': read/write, 'x': read/write/execute}
        """
        return self._mode

    @property
    def node(self):
        """Node of the experiment in the version control tree.

        Value is `None` if the experiment is not connected to the version control tree.

        """
        return self._node

    @property
    def is_done(self):
        """Return True, if this experiment is considered to be finished.

        1. Count how many trials have been completed and compare with ``max_trials``.
        2. Ask ``algorithms`` if they consider there is a chance for further improvement, and
           verify is there is any pending trial.

        .. note::

            To be used as a terminating condition in a ``Worker``.

        """
        trials = self.fetch_trials(with_evc_tree=True)
        num_completed_trials = 0
        num_pending_trials = 0
        for trial in trials:
            if trial.status == "completed":
                num_completed_trials += 1
            elif trial.status in ["new", "reserved", "interrupted"]:
                num_pending_trials += 1

        return (num_completed_trials >= self.max_trials) or (
            self.algorithms.is_done and num_pending_trials == 0
        )

    @property
    def is_broken(self):
        """Return True, if this experiment is considered to be broken.

        Count how many trials are broken and return True if that number has reached
        as given threshold.


        """
        num_broken_trials = self._storage.count_broken_trials(self)
        return num_broken_trials >= self.max_broken

    @property
    def configuration(self):
        """Return a copy of an `Experiment` configuration as a dictionary."""
        config = {}
        for attrname in self.__slots__:
            if attrname.startswith("_"):
                continue
            attribute = copy.deepcopy(getattr(self, attrname))
            config[attrname] = attribute
            if attrname == "space":
                config[attrname] = attribute.configuration
            elif attrname == "algorithms" and not isinstance(attribute, dict):
                config[attrname] = attribute.configuration
            elif attrname == "refers" and isinstance(
                attribute.get("adapter"), BaseAdapter
            ):
                config[attrname]["adapter"] = config[attrname]["adapter"].configuration

        if self.id is not None:
            config["_id"] = self.id

        return copy.deepcopy(config)

    @property
    def stats(self):
        """Calculate :py:class:`orion.core.worker.experiment.ExperimentStats` for this particular
        experiment.
        """
        completed_trials = self.fetch_trials_by_status("completed")

        if not completed_trials:
            return {}
        trials_completed = len(completed_trials)
        best_trials_id = None
        trial = completed_trials[0]
        best_evaluation = trial.objective.value
        best_trials_id = trial.id
        start_time = self.metadata["datetime"]
        finish_time = start_time
        for trial in completed_trials:
            # All trials are going to finish certainly after the start date
            # of the experiment they belong to
            if trial.end_time > finish_time:  # pylint:disable=no-member
                finish_time = trial.end_time
            objective = trial.objective.value
            if objective < best_evaluation:
                best_evaluation = objective
                best_trials_id = trial.id
        duration = finish_time - start_time

        return ExperimentStats(
            trials_completed=trials_completed,
            best_trials_id=best_trials_id,
            best_evaluation=best_evaluation,
            start_time=start_time,
            finish_time=finish_time,
            duration=duration,
        )

    def __repr__(self):
        """Represent the object as a string."""
        return (
            f"Experiment(name={self.name}, metadata.user={self.metadata.get('user', 'n/a')}, "
            f"version={self.version})"
        )
