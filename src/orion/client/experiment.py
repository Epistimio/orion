# pylint:disable=too-many-lines
"""
Experiment wrapper client
=========================

Wraps the core Experiment object to provide further functionalities for the user
"""
from __future__ import annotations

import inspect
import logging
import numbers
import typing
from contextlib import contextmanager
from typing import Callable

import orion.core
from orion.algo.space import Space
from orion.client.runner import Runner, prepare_trial_working_dir
from orion.core.io.database import DuplicateKeyError
from orion.core.utils.exceptions import (
    BrokenExperiment,
    CompletedExperiment,
    ReservationRaceCondition,
    UnsupportedOperation,
    WaitingForTrials,
)
from orion.core.utils.format_trials import dict_to_trial
from orion.core.utils.working_dir import SetupWorkingDir
from orion.core.worker.experiment import AlgoT
from orion.core.worker.producer import Producer
from orion.core.worker.trial import AlreadyReleased, Trial, TrialCM
from orion.core.worker.trial_pacemaker import TrialPacemaker
from orion.executor.base import BaseExecutor, executor_factory
from orion.plotting.base import PlotAccessor
from orion.storage.base import FailedUpdate

if typing.TYPE_CHECKING:
    from orion.core.worker.experiment import Experiment
    from orion.core.worker.experiment_config import ExperimentConfig

log = logging.getLogger(__name__)


def reserve_trial(
    experiment: Experiment,
    producer: Producer,
    pool_size: int,
    timeout: int | None = None,
) -> Trial:
    """Reserve a new trial, or produce and reserve a trial if none are available."""
    log.debug("Trying to reserve a new trial to evaluate.")

    if timeout is not None:
        log.warning(
            "Reservation_timeout is deprecated and will be removed in v0.4.0."
            "Use idle_timeout instead."
        )
    trial = None
    produced = 0

    # Try to reserve an existing trial
    trial = experiment.reserve_trial()

    if trial is None and not (experiment.is_broken or experiment.is_done):
        log.debug("#### Produce new trials.")
        produced = producer.produce(pool_size)
        log.debug("#### %s trials produced.", produced)

        # Try to reverse once more
        trial = experiment.reserve_trial()

    if trial is None:
        if experiment.is_done:
            raise CompletedExperiment()
        elif experiment.is_broken:
            raise BrokenExperiment()
        elif produced == 0:
            raise WaitingForTrials()
        else:
            raise ReservationRaceCondition()

    return trial


# pylint: disable=too-many-public-methods
class ExperimentClient:
    """ExperimentClient providing all functionalities for the python API

    Note that the ExperimentClient is not meant to be instantiated by the user.
    Look at `orion.client.create_experiment` to build an ExperimentClient.

    Parameters
    ----------
    experiment: `orion.core.worker.experiment.Experiment`
        Experiment object serving for interaction with storage
    """

    def __init__(
        self,
        experiment: Experiment[AlgoT],
        executor: BaseExecutor | None = None,
        heartbeat: int | None = None,
    ):
        self._experiment = experiment
        self._producer = Producer(experiment)
        self._pacemakers = {}
        if heartbeat is None:
            heartbeat = orion.core.config.worker.heartbeat
        self.heartbeat = heartbeat

        self._executor = executor
        self._executor_owner = False

        self.plot = PlotAccessor(self)

    ###
    # Attributes
    ###
    @property
    def executor(self):
        """Returns the current executor to use to run jobs in parallel"""
        if self._executor is None:
            self._executor_owner = True
            self._executor = executor_factory.create(
                orion.core.config.worker.executor,
                n_workers=orion.core.config.worker.n_workers,
                **orion.core.config.worker.executor_configuration,
            )

        return self._executor

    @property
    def name(self):
        """Return the name of the experiment in the database."""
        return self._experiment.name

    # pylint: disable=invalid-name
    @property
    def id(self):
        """Return the id of the experiment in the database."""
        return self._experiment.id

    @property
    def version(self):
        """Version of the experiment."""
        return self._experiment.version

    @property
    def max_trials(self):
        """Max-trials to execute before stopping the experiment."""
        return self._experiment.max_trials

    @property
    def max_broken(self):
        """Minimum number of broken trials before the experiment is considered broken."""
        return self._experiment.max_broken

    @property
    def metadata(self):
        """Metadata of the experiment."""
        return self._experiment.metadata

    @property
    def space(self) -> Space:
        """Return problem's parameter `orion.algo.space.Space`."""
        space = self._experiment.space
        assert space is not None
        return space

    @property
    def algorithm(self):
        """Algorithm of the experiment."""
        return self._experiment.algorithm

    @property
    def refers(self):
        """References to the experiment version control"""
        return self._experiment.refers

    @property
    def is_done(self) -> bool:
        """Return True, if this experiment is considered to be finished.

        1. Count how many trials have been completed and compare with `max_trials`.
        2. Ask `algorithms` if they consider there is a chance for further improvement.
        """
        return self._experiment.is_done

    @property
    def is_broken(self):
        """Return True, if this experiment is considered to be broken.

        Count how many trials are broken and return True if that number has reached
        as given threshold.
        """
        return self._experiment.is_broken

    @property
    def configuration(self) -> ExperimentConfig:
        """Return a copy of an `Experiment` configuration as a dictionary."""
        return self._experiment.configuration

    @property
    def stats(self):
        """Calculate :py:class:`orion.core.worker.experiment.ExperimentStats` for this particular
        experiment.
        """
        return self._experiment.stats

    @property
    def node(self):
        """Node of the experiment in the version control tree."""
        return self._experiment.node

    @property
    def working_dir(self):
        """Working directory of the experiment."""
        return self._experiment.working_dir

    @working_dir.setter
    def working_dir(self, value):
        """Working directory of the experiment."""
        self._experiment.working_dir = value

    @property
    def producer(self):
        """Return the producer configuration of the experiment."""
        return self._experiment.producer

    @property
    def mode(self):
        """Return the access right of the experiment

        {'r': read, 'w': read/write, 'x': read/write/execute}
        """
        return self._experiment.mode

    ###
    # Rights
    ###

    def _check_if_writable(self):
        if self.mode == "r":
            calling_function = inspect.stack()[1].function
            raise UnsupportedOperation(
                f"ExperimentClient must have write rights to execute `{calling_function}()`"
            )

    def _check_if_executable(self):
        if self.mode != "x":
            calling_function = inspect.stack()[1].function
            raise UnsupportedOperation(
                f"ExperimentClient must have execution rights to execute `{calling_function}()`"
            )

    ###
    # Queries
    ###

    def to_pandas(self, with_evc_tree=False):
        """Builds a dataframe with the trials of the experiment

        Parameters
        ----------
        with_evc_tree: bool, optional
            Fetch all trials from the EVC tree.
            Default: False

        """
        return self._experiment.to_pandas(with_evc_tree=with_evc_tree)

    def fetch_trials(self, with_evc_tree=False) -> list[Trial]:
        """Fetch all trials of the experiment

        Parameters
        ----------
        with_evc_tree: bool, optional
            Fetch all trials from the EVC tree.
            Default: False

        """
        return self._experiment.fetch_trials(with_evc_tree=with_evc_tree)

    def get_trial(self, trial=None, uid=None):
        """Fetch a single trial

        Parameters
        ----------
        trial: Trial, optional
           trial object to retrieve from the database

        uid: str, optional
            trial id used to retrieve the trial object

        Returns
        -------
        return none if the trial is not found,

        Raises
        ------
        UndefinedCall
            if both trial and uid are not set

        AssertionError
            if both trial and uid are provided and they do not match

        """
        return self._experiment.get_trial(trial, uid)

    def fetch_trials_by_status(self, status, with_evc_tree=False):
        """Fetch all trials with the given status

        Trials are sorted based on ``Trial.submit_time``

        :return: list of :class:`orion.core.worker.trial.Trial` objects
        """
        return self._experiment.fetch_trials_by_status(
            status, with_evc_tree=with_evc_tree
        )

    def fetch_pending_trials(self, with_evc_tree=False):
        """Fetch all trials with status new, interrupted or suspended

        Trials are sorted based on ``Trial.submit_time``

        :return: list of :class:`orion.core.worker.trial.Trial` objects
        """
        return self._experiment.fetch_pending_trials(with_evc_tree=with_evc_tree)

    def fetch_noncompleted_trials(self, with_evc_tree=False):
        """Fetch non-completed trials of this `Experiment` instance.

        Trials are sorted based on ``Trial.submit_time``

        .. note::

            It will return all non-completed trials, including new, reserved, suspended,
            interrupted and broken ones.

        :return: list of non-completed :class:`orion.core.worker.trial.Trial` objects
        """
        return self._experiment.fetch_noncompleted_trials(with_evc_tree=with_evc_tree)

    ###
    # Actions
    ###

    # pylint: disable=unused-argument
    def insert(self, params, results=None, reserve=False):
        """Insert a new trial.

        Experiment must be in writable ('w') or executable ('x') mode.

        Parameters
        ----------
        params: dict
            Parameters of the new trial to add to the database. These parameters
            must comply with the space definition otherwise a ValueError will be raised.
        results: list, optional
            Results to be set for the new trial. Results must have the format
            {name: <str>: type: <'objective', 'constraint' or 'gradient'>, value=<float>} otherwise
            a ValueError will be raised.
            Note that passing results will mark the trial as completed and therefore cannot be
            reserved. The returned trial will have status 'completed'.
            If the results are invalid, the trial will still be inserted but reservation will be
            released.
        reserve: bool, optional
            If reserve=True, the inserted trial will be reserved. `reserve` cannot be True if
            `results` are given.
            Defaults to False.

        Returns
        -------
        `orion.core.worker.trial.Trial`
            The trial inserted in storage. If `reserve=True` and no results are given, the returned
            trial will be in a `reserved` status.

        Raises
        ------
        `ValueError`
            - If results are given and reserve=True
            - If params have invalid format
            - If results have invalid format
        `orion.core.io.database.DuplicateKeyError`
            - If a trial with identical params already exist for the current experiment.
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in writable mode.

        """
        self._check_if_writable()

        if results and reserve:
            raise ValueError(
                "Cannot observe a trial and reserve it. A trial with results has status "
                "`completed` and cannot be reserved."
            )
        trial = dict_to_trial(params, self.space)
        try:
            self._experiment.register_trial(trial, status="reserved")
            self._maintain_reservation(trial)
        except DuplicateKeyError as e:
            message = (
                f"A trial with params {params} already exist for experiment "
                f"{self.name}-v{self.version}"
            )
            raise DuplicateKeyError(message) from e

        if results:
            try:
                self.observe(trial, results)
            except ValueError:
                self._release_reservation(trial)
                raise

            return trial

        if not reserve:
            self.release(trial)

        return trial

    def reserve(self, trial):
        """Reserve a trial.

        Experiment must be in executable ('x') mode.

        Set a trial status to reserve to ensure that concurrent process cannot work on it.
        Trials can only be reserved with status 'new', 'interrupted' or 'suspended'.

        Parameters
        ----------
        trial: `orion.core.worker.trial.Trial`
            Trial to reserve.

        Raises
        ------
        `RuntimeError`
            If trial is reserved by another process
        `ValueError`
            If the trial does not exist in storage.
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        Notes
        -----
        When reserved, a :class:`TrialPacemaker <orion.core.worker.trial_pacemaker.TrialPacemaker>`
        is started to update an heartbeat in storage. The frequency of the heartbeat is configurable
        at creation of experiment or with ``orion.core.config.worker.heartbeat``.
        If the process terminates unexpectedly, the heartbeat will cease and remote processes
        may reset the status of the trial to 'interrupted' when the heartbeat has not been updated
        since twice the value of ``heartbeat``.

        """
        self._check_if_executable()

        if trial.status == "reserved" and trial.id in self._pacemakers:
            log.warning("Trial %s is already reserved.", trial.id)
            return
        elif trial.status == "reserved" and trial.id not in self._pacemakers:
            raise RuntimeError(
                f"Trial {trial.id} is already reserved by another process."
            )
        try:
            self._experiment.set_trial_status(
                trial, "reserved", heartbeat=self.heartbeat
            )
        except FailedUpdate as e:
            if self.get_trial(trial) is None:
                raise ValueError(f"Trial {trial.id} does not exist in database.") from e
            raise RuntimeError(f"Could not reserve trial {trial.id}.") from e

        self._maintain_reservation(trial)

    def release(self, trial, status="interrupted"):
        """Release a trial.

        Release the reservation and stop the heartbeat.

        Experiment must be in writable ('w') or executable ('x') mode.

        Parameters
        ----------
        trial: `orion.core.worker.trial.Trial`
            Trial to reserve.
        status: str, optional
            Set the trial to given status while releasing the reservation.
            Defaults to 'interrupted'.

        Raises
        ------
        `RuntimeError`
            If reservation of the trial has been lost prior to releasing it.
        `AlreadyReleased`
            If reservation of trial was already released
        `ValueError`
            If the trial does not exist in storage.
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in writable mode.

        """
        self._check_if_writable()

        current_status = trial.status
        raise_if_unreserved = True
        try:
            self._experiment.set_trial_status(trial, status, was="reserved")
            self._producer.observe(trial)
        except FailedUpdate as e:
            if self.get_trial(trial) is None:
                raise ValueError(f"Trial {trial.id} does not exist in database.") from e
            if current_status != "reserved":
                raise_if_unreserved = False
                raise AlreadyReleased(
                    f"Trial {trial.id} was already released locally."
                ) from e

            raise RuntimeError(
                f"Reservation for trial {trial.id} has been lost before release."
            ) from e
        finally:
            self._release_reservation(trial, raise_if_unreserved=raise_if_unreserved)

    def suggest(self, pool_size=0):
        """Suggest a trial to execute.

        Experiment must be in executable ('x') mode.

        If any trial is available (new or interrupted), it selects one and reserves it.
        Otherwise, the algorithm is used to generate a new trial that is registered in storage and
        reserved.

        Parameters
        ----------
        pool_size: int, optional
            Number of trials to sample at a time. If 0, default to global config if defined,
            else 1.  Increase it to improve the sampling speed if workers spend too much time
            waiting for algorithms to sample points. An algorithm will try sampling `pool_size`
            trials but may return less. Note: The method will still return only 1 trial even though
            if the pool size is larger than 1. This is because atomic reservation of trials
            can only be done one at a time.

        Returns
        -------
        `orior.core.worker.trial.Trial`
            Reserved trial for execution.

        Raises
        ------
        :class:`orion.core.utils.exceptions.WaitingForTrials`
            if the experiment is not completed and algorithm needs to wait for some
            trials to complete before it can suggest new trials.

        :class:`orion.core.utils.exceptions.BrokenExperiment`
            if too many trials failed to run and the experiment cannot continue.
            This is determined by ``max_broken`` in the configuration of the experiment.

        :class:`orion.core.utils.exceptions.ReservationRaceCondition`
            If a trial could not be reserved right after they were generated

        :class:`orion.core.utils.exceptions.CompletedExperiment`
            if the experiment was completed and algorithm could not sample new trials.

        :class:`orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        """
        self._check_if_executable()
        if not pool_size:
            pool_size = orion.core.config.worker.pool_size
        if not pool_size:
            pool_size = 1

        if self.is_broken:
            raise BrokenExperiment("Trials failed too many times")

        if self.is_done:
            raise CompletedExperiment("Experiment is done, cannot sample more trials.")

        try:
            trial = reserve_trial(self._experiment, self._producer, pool_size)

        except (ReservationRaceCondition, WaitingForTrials) as e:
            if self.is_broken:
                raise BrokenExperiment("Trials failed too many times") from e

            raise e

        # This is to handle cases where experiment was completed during call to `reserve_trial`
        if trial is None and self.is_done:
            raise CompletedExperiment("Producer is done, cannot sample more trials.")
        elif trial is None and self.is_broken:
            raise BrokenExperiment("Trials failed too many times")

        self._maintain_reservation(trial)
        return TrialCM(self, trial)

    def observe(
        self,
        trial: Trial,
        results: list[dict] | float,
        name: str = "objective",
    ) -> None:
        """Observe trial results

        Experiment must be in executable ('x') mode.

        Parameters
        ----------
        trial: `orion.core.worker.trial.Trial`
            Reserved trial to observe.
        results: list or float
            Results to be set for the new trial. Results must have the format
            {name: <str>: type: <'objective', 'constraint' or 'gradient'>, value=<float>} otherwise
            a ValueError will be raised. If the results are invalid, the trial will not be released.
            If `results` is a float, the result type will be 'objective'.
        name: str
            Name of the result if `results` is a float. Default: 'objective'.

        Returns
        -------
        `orion.core.worker.trial.Trial`
            The trial inserted in storage. If `reserve=True` and no results are given, the returned
            trial will be in a `reserved` status.

        Raises
        ------
        `ValueError`
            - If results have invalid format
            - If the trial does not exist in storage.
        `RuntimeError`
            If reservation of the trial has been lost prior to releasing it.
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.
        `orion.core.utils.exceptions.InvalidResult`
            If the format of trial result is invalid.
        """
        self._check_if_executable()

        if isinstance(results, numbers.Number):
            results = [dict(value=results, name=name, type="objective")]

        trial.results += [Trial.Result(**result) for result in results]
        raise_if_unreserved = True
        try:
            self._experiment.update_completed_trial(trial)
            self._producer.observe(trial)
        except FailedUpdate as e:
            if self.get_trial(trial) is None:
                raise_if_unreserved = False
                raise ValueError(f"Trial {trial.id} does not exist in database.") from e

            raise RuntimeError(
                f"Reservation for trial {trial.id} has been lost."
            ) from e
        finally:
            self._release_reservation(trial, raise_if_unreserved=raise_if_unreserved)

    @contextmanager
    def tmp_executor(self, executor, **config):
        """Temporarily change the executor backend of the experiment client.

        Parameters
        ----------
        executor: str or :class:`orion.executor.base.BaseExecutor`
            The executor to use. If it is a ``str``, the provided ``config`` will be used
            to create the executor with ``executor_factory.create(executor, **config)``.
        **config:
            Configuration to use if ``executor`` is a ``str``.

        """
        if isinstance(executor, str):
            executor = executor_factory.create(executor, **config)
        old_executor = self._executor
        self._executor = executor
        with executor:
            yield self
        self._executor = old_executor

    # pylint:disable=too-many-arguments
    def workon(
        self,
        fct: Callable,
        n_workers: int | None = None,
        pool_size: int = 0,
        max_trials: int | None = None,
        max_trials_per_worker: int | None = None,
        max_broken: int | None = None,
        trial_arg: str | None = None,
        on_error: Callable[[ExperimentClient, Exception, int], bool] | None = None,
        prepare_trial: Callable[
            [ExperimentClient, Trial], None
        ] = prepare_trial_working_dir,
        idle_timeout: int | None = None,
        **kwargs,
    ) -> int:
        """Optimize a given function

        Experiment must be in executable ('x') mode.

        Parameters
        ----------
        fct: callable
            Function to optimize. Must take arguments provided by trial.params. Additional constant
            parameter can be passed as ``**kwargs`` to `workon`. Function must return the final
            objective.
        n_workers: int, optional
            Number of workers to run in parallel. Defaults to value of global config.
        pool_size: int, optional
            Number of trials to sample at a time. If 0, defaults to `n_workers` or value of global
            config if defined.  Increase it to improve the sampling speed if workers spend too much
            time waiting for algorithms to sample points. An algorithm will try sampling
            `pool_size` trials but may return less.
        max_trials: int, optional
            Maximum number of trials to execute within ``workon``. If the experiment or algorithm
            reach status is_done before, the execution of ``workon`` terminates.
            Defaults to experiment's max trial. If ``max_trials`` is larger than
            ``experiment.max_trials``, the experiment will stop when reaching
            ``experiment.max_trials``.
        max_trials_per_worker: int, optional
            Maximum number of trials to execute within each worker. ``max_trials`` and
            ``experiment.max_trials`` have precedence. Defaults to global config.
        max_broken: int, optional
            Maximum number of broken trials to accept during `workon`. When this threshold is
            reached the function will raise :class:`orion.core.utils.exceptions.BrokenExperiment`.
            Defaults to value of global config.
        trial_arg: str, optional
            The function ``fct`` may support receiving the trial as an argument. This argument name
            can be specified with ``trial_arg``. If not defined (``None``), then only the
            hyperparameters will be passed to `fct`.
        on_error: callable, optional
            Callback that is executed if an error occur during the execution of ``fct``.
            The signature of the callback must be
            ``foo(ExperimentClient, Trial, Error, nb_of_worker_broken_trials)``.
            If the callblack returns False, the error will be ignored, otherwise it is counted
            for the threshold `max_broken`. In case of critical errors, you may also directly
            raise an error and force break out of ``workon``.
        prepare_trial: callable, optional
            Callback that is executed before the trial is submitted to workers for execution.
            Default is `orion.client.runner.prepare_trial_working_dir` which will create working
            directory of trials if necessary.
        idle_timeout: int, optional
            Maximum time (seconds) allowed for idle workers. LazyWorkers will be raised if
            timeout is reached. Such timeout are generally caused when reaching the
            end of the optimization when no new trials can be sampled for the idle workers.
            Defaults to ``orion.core.config.worker.idle_timeout``.
        **kwargs
            Constant argument to pass to `fct` in addition to trial.params. If values in kwargs are
            present in trial.params, the latter takes precedence.

        Raises
        ------
        :class:`orion.core.utils.exceptions.LazyWorkers`
             If workers stay idle for too long.

        :class:`orion.core.utils.exceptions.InvalidResult`
             If results returned by `fct` have invalid format.

        :class:`orion.core.utils.exceptions.WaitingForTrials`
            If the experiment is not completed and algorithm needs to wait for some
            trials to complete before it can suggest new trials.

        :class:`orion.core.utils.exceptions.BrokenExperiment`
            If too many trials failed to run and the experiment cannot continue.
            This is determined by ``max_broken`` in the configuration of the experiment.

        :class:`orion.core.utils.exceptions.ReservationRaceCondition`
            If a trial could not be reserved right after they were generated.

        :class:`orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        """
        self._check_if_executable()

        if n_workers is None:
            n_workers = self.executor.n_workers
        else:
            if n_workers > self.executor.n_workers:
                log.warning(
                    "The required number of workers %s is bigger than executor configuration %s",
                    str(n_workers),
                    str(self.executor.n_workers),
                )

        if not pool_size:
            pool_size = orion.core.config.worker.pool_size
        if not pool_size:
            pool_size = n_workers

        if not idle_timeout:
            idle_timeout = orion.core.config.worker.idle_timeout

        if max_trials is None:
            max_trials = self.max_trials

        if max_trials_per_worker is None:
            max_trials_per_worker = orion.core.config.worker.max_trials

        if max_broken is None:
            max_broken = orion.core.config.worker.max_broken

        # Use worker's max_trials inside `exp.is_done` to reduce chance of
        # race condition for trials creation
        assert self.max_trials is not None
        assert max_trials is not None

        if self.max_trials > max_trials:
            self._experiment.max_trials = max_trials
            assert self._experiment.algorithm is not None
            self._experiment.algorithm.max_trials = max_trials

        with SetupWorkingDir(self):
            runner = Runner(
                self,
                fct,
                pool_size=pool_size,
                idle_timeout=idle_timeout,
                max_trials_per_worker=max_trials_per_worker,
                max_broken=max_broken,
                trial_arg=trial_arg,
                on_error=on_error,
                prepare_trial=prepare_trial,
                n_workers=n_workers,
                **kwargs,
            )

            rval = runner.run()

        return rval

    def close(self):
        """Verify that no reserved trials are remaining.

        Experiment must be in executable ('x') mode.

        Raises
        ------
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        """
        self._check_if_executable()
        self._free_executor()

        if self._pacemakers:
            raise RuntimeError(
                f"There is still reserved trials: {self._pacemakers.keys()}\n"
                "Release all trials before closing the client, using "
                "client.release(trial)."
            )

    ###
    # Private
    ###
    def __del__(self):
        self._free_executor()

    def _free_executor(self):
        if self._executor_owner:
            self._executor.__exit__(None, None, None)
            self._executor = None
            self._executor_owner = False

    def __repr__(self):
        """Represent the object as a string."""
        return f"Experiment(name={self.name}, version={self.version})"

    def _verify_reservation(self, trial):
        if trial.id not in self._pacemakers:
            raise RuntimeError(
                f"Trial {trial.id} had no pacemakers. Was it reserved properly?"
            )

        if self.get_trial(trial).status != "reserved":
            self._release_reservation(trial)
            raise RuntimeError(f"Reservation for trial {trial.id} has been lost.")

    def _maintain_reservation(self, trial):
        self._pacemakers[trial.id] = TrialPacemaker(trial, self.storage)
        self._pacemakers[trial.id].start()

    def _release_reservation(self, trial, raise_if_unreserved=True):
        if trial.id not in self._pacemakers:
            if raise_if_unreserved:
                raise RuntimeError(
                    f"Trial {trial.id} had no pacemakers. Was it reserved properly?"
                )
            else:
                return

        self._pacemakers.pop(trial.id).stop()

    @property
    def storage(self):
        """Return the storage currently in use by this client"""
        return self._experiment.storage
