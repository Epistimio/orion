"""Implement a ExperimentClient but that go through
a REST API instead of instantiating its own hyperparameter optimizer

"""

import numbers

import orion.core
from orion.client.experiment import ExperimentClient
from orion.core.utils.exceptions import BrokenExperiment
from orion.core.worker.trial import TrialCM
from orion.plotting.base import PlotAccessor
from orion.service.client.actions import ClientActionREST
from orion.service.client.workon import WorkonClientREST


# pylint: disable=too-many-public-methods
class ExperimentClientREST(ExperimentClient):
    """Minimal REST API, only implements the required method to run workon.

    The REST API is split two:

    * Workon API, processing heavy - low IO
    * Storage API, low processing - IO heavy

    In production the APIs are handled by different hosts.
    The split in the code is made so both can be tested separately
    and make sure outage in one does not impact the other

    Notes
    -----
    The main difference between the ``ExperimentClient`` and
    the REST client is that the algorithm is not running alongside the client;
    instead it is ran on the server.

    So on this client there are no (Trial) Producer, instead it relies on the rest API to suggest
    the trials.

    The client is composed of two main objects;
    the REST client which is in charge of communicating with
    the algo running remotely (suggest & observe);
    and the REST storage which is in charge of fetching information.

    The REST storage is mostly read only as we should not modify
    the experiment state while it is running.

    To achieve this, the REST client overrides some selected methods from
    the ExperimentClient which short-circuits
    the execution of the algorithm to use the rest calls.

    """

    # pylint: disable=too-many-arguments,unused-argument
    @staticmethod
    def create_experiment(
        name,
        version=None,
        space=None,
        algorithms=None,
        strategy=None,
        max_trials=None,
        max_broken=None,
        storage=None,
        branching=None,
        max_idle_time=None,
        heartbeat=None,
        working_dir=None,
        debug=False,
        knowledge_base=None,
        executor=None,
    ):
        """Instantiate an experiment using the REST API instead of relying on local storage"""

        endpoint, token = storage["endpoint"], storage["token"]

        workon = WorkonClientREST(endpoint, token)

        experiment = workon.new_experiment(
            name,
            version=version,
            space=space,
            algorithms=algorithms,
            strategy=strategy,
            max_trials=max_trials,
            max_broken=max_broken,
            branching=branching,
            max_idle_time=max_idle_time,
            heartbeat=heartbeat,
            working_dir=working_dir,
            debug=debug,
        )

        client = ExperimentClientREST(
            experiment,
            executor=executor,
            heartbeat=heartbeat,
        )

        storage = ClientActionREST(experiment, endpoint, token)

        client.workon_client = workon
        client.storage_client = storage
        return client

    # pylint: disable=super-init-not-called
    def __init__(self, experiment, executor=None, heartbeat=None):
        # Do not call super here; we do not want to instantiate the producer
        self.workon_client = None

        if heartbeat is None:
            heartbeat = orion.core.config.worker.heartbeat

        self._experiment = experiment
        self.heartbeat = heartbeat
        self._executor = executor
        self._executor_owner = False
        self.storage_client = None
        self.plot = PlotAccessor(self)

    def to_pandas(self, with_evc_tree=False):
        raise RuntimeError()

    @property
    def _pacemakers(self):
        # pylint: disable=protected-access
        return self.workon_client._pacemakers

    #
    # Workon REST API overrides
    #

    @property
    def is_broken(self):
        """See `~ExperimentClient.is_broken`"""
        try:
            self.is_done
        except BrokenExperiment:
            return True
        return False

    @property
    def is_done(self):
        """See `~ExperimentClient.is_done`"""
        return self.workon_client.is_done()

    def suggest(self, pool_size=0) -> TrialCM:
        """See `~ExperimentClient.suggest`"""
        remote_trial = self.workon_client.suggest(pool_size=pool_size)
        return remote_trial

    def observe(self, trial, results, name: str = "objective"):
        """See `~ExperimentClient.observe`"""

        if isinstance(results, numbers.Number):
            results = [dict(value=results, name=name, type="objective")]

        self.workon_client.observe(trial, results)

    def release(self, trial, status="interrupted"):
        """See `~ExperimentClient.release`"""
        # we probably should not expose this
        return self.workon_client.release(trial, status)

    #
    # Disabled
    #

    def reserve(self, trial):
        """See `~ExperimentClient.reserve`"""
        raise NotImplementedError("REST API reserve on suggest")

    @property
    def producer(self):
        """Return the producer configuration of the experiment."""
        raise NotImplementedError("REST API does not instantiate the producer")

    @property
    def storage(self):
        """Return the storage currently in use by this client"""
        raise NotImplementedError("REST API does not have a storage")

    #
    # Storage REST API
    #

    def insert(self, params, results=None, reserve=False):
        """See `~ExperimentClient.insert`

        Cannot reserve a trial with the REST API

        """
        return self.storage_client.insert(params, results, False)

    def fetch_noncompleted_trials(self, with_evc_tree=False):
        """See `~ExperimentClient.fetch_noncompleted_trials`"""
        return self.storage_client.fetch_noncompleted_trials(with_evc_tree)

    def fetch_pending_trials(self, with_evc_tree=False):
        """See `~ExperimentClient.fetch_pending_trials`"""
        return self.storage_client.fetch_pending_trials(with_evc_tree)

    def fetch_trials_by_status(self, status, with_evc_tree=False):
        """See `~ExperimentClient.fetch_trials_by_status`"""
        return self.storage_client.fetch_trials_by_status(status, with_evc_tree)

    def get_trial(self, trial=None, uid=None):
        """See `~ExperimentClient.get_trial`"""
        return self.storage_client.get_trial(trial, uid)

    def fetch_trials(self, with_evc_tree=False):
        """See `~ExperimentClient.fetch_trials`"""
        return self.storage_client.fetch_trials(with_evc_tree)

    def close(self):
        self._free_executor()
        self.workon_client.close()
