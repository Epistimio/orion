"""
Generic Storage Protocol
========================

Implement a generic protocol to allow Orion to communicate using different storage backend.

Storage protocol is a generic way of allowing Orion to interface with different storage.
MongoDB, track, cometML, MLFLow, etc...

Examples
--------
>>> storage_factory.create('track', uri='file://orion_test.json')
>>> storage_factory.create('legacy', experiment=...)

Notes
-----
When retrieving an already initialized Storage object you should use `get_storage`.
`storage_factory.create()` should only be used for initialization purposes as `get_storage`
raises more granular error messages.

"""
import contextlib
import copy
import logging

import orion.core
from orion.core.io import resolve_config
from orion.core.utils import GenericFactory

log = logging.getLogger(__name__)


def get_uid(item=None, uid=None, force_uid=True):
    """Return uid either from `item` or directly uid.

    Parameters
    ----------
    item: Experiment or Trial, optional
       Object with .id attribute

    uid: str, optional
        str id representation

    force_uid: bool, optional
        If True, at least one of item or uid must be passed.

    Raises
    ------
    UndefinedCall
        if both item and uid are not set and force_uid is True

    AssertionError
        if both item and uid are provided and they do not match
    """
    if item is not None and uid is not None:
        assert item.id == uid

    if uid is None:
        if item is None and force_uid:
            raise MissingArguments("Either `item` or `uid` should be set")
        elif item is not None:
            uid = item.id

    return uid


def get_trial_uid_and_exp(trial=None, uid=None, experiment_uid=None):
    """Return trial and experiment uid either from `trial` or directly uids.

    Parameters
    ----------
    trial: Trial, optional
       Object with .id attribute

    uid: str, optional
        str id representation of the trial

    experiment_uid: str, optional
        str id representation of the experiment

    Raises
    ------
    UndefinedCall
        if both trial and (uid or experiment_uid) are not set

    AssertionError
        if both trial and (uid or experiment_uid) are provided and they do not match

    Returns
    -------
    (trial uid, experiment uid)
    """

    if trial is None and experiment_uid is None:
        raise MissingArguments(
            "Either `trial` or (`uid` and `experiment_uid`) should be set"
        )

    if trial is not None and experiment_uid:
        assert trial.experiment == experiment_uid
    elif trial is not None:
        experiment_uid = trial.experiment

    trial_uid = get_uid(trial, uid)

    return trial_uid, experiment_uid


class FailedUpdate(Exception):
    """Exception raised when we are unable to update a trial' status"""


class MissingArguments(Exception):
    """Raised when calling a function without the minimal set of parameters"""


class LockAcquisitionTimeout(Exception):
    """Raised when the lock acquisition timeout (not lock is granted)."""


class LockedAlgorithmState:
    """Locked state of the algorithm from the storage.

    This class helps handle setting the state of the algorithm or resetting it in case
    the execution crashes during the lock.

    Parameters
    ----------
    state: dict
        Dictionary representing the state of the algorithm.
    configuration: dict
        Configuration of the locked algorithm.
    locked: bool
        Whether the algorithm is locked or not. Default: True
    """

    def __init__(self, state, configuration, locked=True):
        self._original_state = state
        self.configuration = configuration
        self._state = state
        self.locked = locked

    @property
    def state(self):
        """State of the algorithm"""
        return self._state

    def set_state(self, state):
        """Update the state of the algorithm that should be saved back in storage."""
        self._state = state

    def reset(self):
        """Set back algorithm state to original state found in storage."""
        self._state = self._original_state


class BaseStorageProtocol:
    """Implement a generic protocol to allow Orion to communicate using
    different storage backend

    """

    def create_benchmark(self, config):
        """Insert a new benchmark inside the database"""
        raise NotImplementedError()

    def fetch_benchmark(self, query, selection=None):
        """Fetch all benchmarks that match the query"""
        raise NotImplementedError()

    def create_experiment(self, config):
        """Insert a new experiment inside the database"""
        raise NotImplementedError()

    def delete_experiment(self, experiment=None, uid=None):
        """Delete matching experiments from the database

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database

        uid: str, optional
            experiment id used to retrieve the trial object

        Returns
        -------
        Number of experiments deleted.

        Raises
        ------
        UndefinedCall
            if both experiment and uid are not set

        AssertionError
            if both experiment and uid are provided and they do not match
        """
        raise NotImplementedError()

    def update_experiment(self, experiment=None, uid=None, where=None, **kwargs):
        """Update the fields of a given experiment

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database

        uid: str, optional
            experiment id used to retrieve the trial object

        where: Optional[dict]
            constraint experiment must respect

        **kwargs: dict
            a dictionary of fields to update

        Returns
        -------
        returns true if the underlying storage was updated

        Raises
        ------
        UndefinedCall
            if both experiment and uid are not set

        AssertionError
            if both experiment and uid are provided and they do not match

        """
        raise NotImplementedError()

    def fetch_experiments(self, query, selection=None):
        """Fetch all experiments that match the query"""
        raise NotImplementedError()

    def register_trial(self, trial):
        """Create a new trial to be executed"""
        raise NotImplementedError()

    def delete_trials(self, experiment=None, uid=None, where=None):
        """Delete matching trials from the database

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database

        uid: str, optional
            experiment id used to retrieve the trial object

        where: Optional[dict]
            constraint trials must respect

        Returns
        -------
        Number of trials deleted.

        Raises
        ------
        UndefinedCall
            if both experiment and uid are not set

        AssertionError
            if both experiment and uid are provided and they do not match
        """
        raise NotImplementedError()

    def reserve_trial(self, experiment):
        """Select a pending trial and reserve it for the worker

        Returns
        -------
        Returns the reserved trial or None if no trials were found

        """
        raise NotImplementedError()

    def fetch_trials(self, experiment=None, uid=None, where=None):
        """Fetch all the trials of an experiment in the database

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database

        uid: str, optional
            experiment id used to retrieve the trial object

        where: Optional[dict]
            constraint trials must respect

        Returns
        -------
        return none if the experiment is not found,

        Raises
        ------
        UndefinedCall
            if both experiment and uid are not set

        AssertionError
            if both experiment and uid are provided and they do not match

        """
        raise NotImplementedError()

    def update_trials(self, experiment=None, uid=None, where=None, **kwargs):
        """Update trials of a given experiment matching a query

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database

        uid: str, optional
            experiment id used to retrieve the trial object

        where: Optional[dict]
            constraint trials must respect

        **kwargs: dict
            a dictionary of fields to update

        Raises
        ------
        UndefinedCall
            if both experiment and uid are not set

        AssertionError
            if both experiment and uid are provided and they do not match

        """
        raise NotImplementedError()

    def update_trial(
        self, trial=None, uid=None, experiment_uid=None, where=None, **kwargs
    ):
        """Update fields of a given trial

        Parameters
        ----------
        trial: Trial, optional
           trial object to update in the database

        uid: str, optional
            id of the trial to update in the database

        experiment_uid: str, optional
            experiment id of the trial to update in the database

        where: Optional[dict]
            constraint trials must respect. Note: useful to handle race conditions.

        **kwargs: dict
            a dictionary of fields to update

        Raises
        ------
        UndefinedCall
            if both trial and uid are not set

        AssertionError
            if both trial and uid are provided and they do not match

        """
        raise NotImplementedError()

    def get_trial(self, trial=None, uid=None, experiment_uid=None):
        """Fetch a single trial

        Parameters
        ----------
        trial: Trial, optional
           trial object to retrieve from the database

        uid: str, optional
            trial id used to retrieve the trial object

        experiment_uid: str, optional
            experiment id used to retrieve the trial object

        Returns
        -------
        return None if the trial is not found,

        Raises
        ------
        UndefinedCall
            if both trial and uid are not set

        AssertionError
            if both trial and uid are provided and they do not match

        """
        raise NotImplementedError()

    def fetch_lost_trials(self, experiment):
        """Fetch all trials that have a heartbeat older than
        some given time delta (2 minutes by default)
        """
        raise NotImplementedError()

    def retrieve_result(self, trial, *args, **kwargs):
        """Fetch the result from a given medium (file, db, socket, etc..) for a given trial and
        insert it into the trial object
        """
        raise NotImplementedError()

    def push_trial_results(self, trial):
        """Push the trial's results to the database"""
        raise NotImplementedError()

    def set_trial_status(self, trial, status, heartbeat=None, was=None):
        """Update the trial status and the heartbeat

        Parameters
        ----------
        trial: `Trial` object
            Trial object to update in the database.
        status: str
            Status to be set to the trial
        heartbeat: datetime, optional
            New heartbeat to update simultaneously with status
        was: str, optional
            The status the trial should be set to in the database.
            If None, current ``trial.status`` will be used.
            This is used to ensure coherence in the database, protecting
            against race conditions for instance.

        Raises
        ------
        FailedUpdate
            The exception is raised if the status of the trial object
            does not match the status in the database

        """
        raise NotImplementedError()

    def fetch_pending_trials(self, experiment):
        """Fetch all trials that are available to be executed by a worker,
        this includes new, suspended and interrupted trials
        """
        raise NotImplementedError()

    def fetch_noncompleted_trials(self, experiment):
        """Fetch all non completed trials"""
        raise NotImplementedError()

    def fetch_trials_by_status(self, experiment, status):
        """Fetch all trials with the given status"""
        raise NotImplementedError()

    def count_completed_trials(self, experiment):
        """Count the number of completed trials"""
        raise NotImplementedError()

    def count_broken_trials(self, experiment):
        """Count the number of broken trials"""
        raise NotImplementedError()

    def update_heartbeat(self, trial):
        """Update trial's heartbeat"""
        raise NotImplementedError()

    def initialize_algorithm_lock(self, experiment_id, algorithm_config):
        """Initialize algorithm lock for given experiment

        Parameters
        ----------
        experiment_id: int or str
            ID of the experiment in storage.
        algorithm_config: dict
            Configuration of the algorithm.
        """
        raise NotImplementedError()

    def release_algorithm_lock(self, experiment=None, uid=None, new_state=None):
        """Release the algorithm lock

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database
        uid: str, optional
            experiment id used to retrieve the trial object.
        new_state: dict, optional
             The new state of the algorithm that should be saved in the lock object.
             If None, the previous state is preserved in the lock object in storage.
        """
        raise NotImplementedError()

    def get_algorithm_lock_info(self, experiment=None, uid=None):
        """Load algorithm lock info

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database
        uid: str, optional
            experiment id used to retrieve the trial object.

        Returns
        -------
        ``orion.storage.base.LockedAlgorithmState``
            The locked state of the algorithm. Note that the lock is not acquired by the process
            calling ``get_algorithm_lock_info`` and the value of LockedAlgorithmState.locked
            may not be valid if another process is running and could acquire the lock concurrently.
        """
        raise NotImplementedError()

    def delete_algorithm_lock(self, experiment=None, uid=None):
        """Delete experiment algorithm lock from the storage

        Parameters
        ----------
        experiment: Experiment, optional
           experiment object to retrieve from the database
        uid: str, optional
            experiment id used to retrieve the trial object

        Returns
        -------
        Number of algorithm lock deleted. Should 1 if successful, 0 is failed.

        Raises
        ------
        UndefinedCall
            if both experiment and uid are not set
        AssertionError
            if both experiment and uid are provided and they do not match
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def acquire_algorithm_lock(self, experiment, timeout=600, retry_interval=1):
        """Acquire lock on algorithm in storage

        This method is a contextmanager and should be called using the ``with``-clause.

        Parameters
        ----------
        experiment: Experiment
           experiment object to retrieve from the storage
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
        ``orion.storage.base.LockAcquisitionTimeout``
            The lock could not be obtained in less than ``timeout`` seconds.
        """
        raise NotImplementedError()


storage_factory = GenericFactory(BaseStorageProtocol)


def setup_storage(storage=None, debug=False):
    """Create the storage instance from a configuration.

    Parameters
    ----------
    config: dict, optional
        Configuration for the storage backend. If not defined, global configuration
        is used.
    debug: bool, optional
        If using in debug mode, the storage config is overridden with legacy:EphemeralDB.
        Defaults to False.

    """
    if storage is None:
        storage = orion.core.config.storage.to_dict()

    storage = copy.deepcopy(storage)

    if storage.get("type") == "legacy" and "database" not in storage:
        storage["database"] = orion.core.config.storage.database.to_dict()
    elif storage.get("type") is None and "database" in storage:
        storage["type"] = "legacy"

    # If using same storage type
    if storage["type"] == orion.core.config.storage.type:
        storage = resolve_config.merge_configs(
            orion.core.config.storage.to_dict(), storage
        )

    if debug:
        storage = {"type": "legacy", "database": {"type": "EphemeralDB"}}

    storage_type = storage.pop("type")

    log.debug("Creating %s storage client with args: %s", storage_type, storage)
    try:
        return storage_factory.create(of_type=storage_type, **storage)
    except ValueError:
        if storage_factory.create().__class__.__name__.lower() != storage_type.lower():
            raise


# pylint: disable=too-few-public-methods
class ReadOnlyStorageProtocol:
    """Read-only interface from a storage protocol.

    .. seealso::

        :py:class:`BaseStorageProtocol`
    """

    __slots__ = ("_storage",)
    valid_attributes = {
        "get_trial",
        "fetch_trials",
        "fetch_experiments",
        "count_broken_trials",
        "count_completed_trials",
        "fetch_noncompleted_trials",
        "fetch_pending_trials",
        "fetch_lost_trials",
        "fetch_trials_by_status",
    }

    def __init__(self, protocol):
        """Init method, see attributes of :class:`BaseStorageProtocol`."""
        self._storage = protocol

    def __getattr__(self, attr):
        """Get attribute only if valid"""
        if attr not in self.valid_attributes:
            raise AttributeError(
                "Cannot access attribute %s on ReadOnlyStorageProtocol." % attr
            )

        return getattr(self._storage, attr)
