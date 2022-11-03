"""
Python API
==========

Provides functions for communicating with `orion.core`.

"""
from __future__ import annotations

import logging
import typing
from typing import Any, Callable

# pylint: disable=consider-using-from-import
import orion.core.io.experiment_builder as experiment_builder
from orion.algo.base import BaseAlgorithm
from orion.client.cli import (
    interrupt_trial,
    report_bad_trial,
    report_objective,
    report_results,
)
from orion.client.experiment import ExperimentClient
from orion.core.utils.exceptions import RaceCondition
from orion.core.worker.producer import Producer
from orion.core.worker.warm_start.knowledge_base import KnowledgeBase
from orion.executor.base import BaseExecutor
from orion.storage.base import BaseStorageProtocol, setup_storage

__all__ = [
    "interrupt_trial",
    "report_bad_trial",
    "report_objective",
    "report_results",
    "create_experiment",
    "build_experiment",
    "get_experiment",
    "workon",
]

log = logging.getLogger(__name__)


def create_experiment(name: str, **config):
    """Build an experiment to be executable

    This function is deprecated and will be removed in v0.3.0. Use `build_experiment`
    instead.
    """
    return build_experiment(name, **config)


# pylint: disable=too-many-arguments
def build_experiment(
    name: str,
    version: int | None = None,
    space: dict[str, Any] | None = None,
    algorithm: type[BaseAlgorithm] | dict | None = None,
    algorithms: type[BaseAlgorithm] | dict | None = None,
    strategy: str | dict | None = None,
    max_trials: int | None = None,
    max_broken: int | None = None,
    storage: dict | BaseStorageProtocol | None = None,
    branching: dict | None = None,
    max_idle_time: int | None = None,
    heartbeat: int | None = None,
    working_dir: str | None = None,
    debug: bool = False,
    knowledge_base: KnowledgeBase | dict | None = None,
    executor: BaseExecutor | None = None,
) -> ExperimentClient:
    """Build an experiment to be executable

    Building the experiment can result in branching if there are any changes in the environment.
    This is required to ensure coherence between execution of trials. For an experiment
    in read/write mode without execution rights, see `get_experiment`.

    There is 2 main scenarios

    1) The experiment is new

    ``name`` and ``space`` arguments are required, otherwise ``NoConfigurationError`` will be
    raised.

    All other arguments (``algorithm``, ``strategy``, ``max_trials``, ``storage``, ``branching``
    and ``working_dir``) will be replaced by system's defaults if omitted. The system's defaults can
    also be overridden in global configuration file as described for the database in
    :ref:`Database Configuration`. We do not recommend overriding the algorithm configuration using
    system's default, but overriding the storage configuration can be very convenient if the same
    storage is used for all your experiments.

    2) The experiment exist in the database.

    We can break down this scenario in two sub-scenarios for clarity.

    2.1) Only experiment name is given.

    The configuration will be fetched from database.

    2.2) Some other arguments than the name are given.

    The configuration will be fetched from database and given arguments will override them.
    ``max_trials`` and ``max_broken`` may be overwritten in DB, but any other changes will lead to a
    branching. Instead of creating the experiment ``(name, version)``, it will create a new
    experiment ``(name, version+1)`` which will have the same configuration than ``(name, version)``
    except for the differing arguments given by user. This new experiment will have access to trials
    of ``(name, version)``, adapted according to the differences between ``version`` and
    ``version+1``.  A previous version can be accessed by specifying the ``version`` argument.

    Causes of experiment branching are:

    - Change of search space

        - New dimension

        - Different prior

        - Missing dimension

    - Change of algorithm

    - Change of strategy (Not implemented yet)

    - Change of code version (Only supported by commandline API for now)

    - Change of orion version

    Parameters
    ----------
    name: str
        Name of the experiment
    version: int, optional
        Version of the experiment. Defaults to last existing version for a given ``name``
        or 1 for new experiment.
    space: dict, optional
        Optimization space of the algorithm. Should have the form ``dict(name='<prior>(args)')``.
    algorithm: str or dict, optional
        Algorithm used for optimization.
    strategy: str or dict, optional
        Deprecated and will be remove in v0.4. It should now be set in algorithm configuration
        directly if it supports it.
    max_trials: int, optional
        Maximum number or trials before the experiment is considered done.
    max_broken: int, optional
        Number of broken trials for the experiment to be considered broken.
    storage: dict or BaseStorageProtocol, optional
        Configuration of the storage backend.
    working_dir: str, optional
        Working directory created for the experiment inside which a unique folder will be created
        for each trial. Defaults to a temporary directory that is deleted at end of execution.
    max_idle_time: int, optional
        Deprecated and will be removed in v0.3.0.
        Use experiment.workon(idle_timeout) instead.
    heartbeat: int, optional
        Frequency (seconds) at which the heartbeat of the trial is updated.
        If the heartbeat of a `reserved` trial is larger than twice the configured
        heartbeat, Oríon will reset the status of the trial to `interrupted`.
        This allows restoring lost trials (ex: due to killed worker).
        Defaults to ``orion.core.config.worker.max_idle_time``.
    debug: bool, optional
        If using in debug mode, the storage config is overridden with legacy:EphemeralDB.
        Defaults to False.
    branching: dict, optional
        Arguments to control the branching.

        branch_to: str, optional
            Name of the experiment to branch to. The parent experiment will be the one specified by
            ``(name, version)``, and the child will be ``(branch_to, 1)``.
        branch_from: str, optional
            Name of the experiment to branch from.
            The parent experiment will be the one specified by
            ``(branch_from, last version)``, and the child will be ``(name, 1)``.
        manual_resolution: bool, optional
            Starts the prompt to resolve manually the conflicts. Defaults to False.
        algorithm_change: bool, optional
            Whether to automatically solve the algorithm conflict (change of algo config).
            Defaults to True.
        orion_version_change: bool, optional
            Whether to automatically solve the orion version conflict.
            Defaults to True.
        code_change_type: str, optional
            How to resolve code change automatically. Must be one of 'noeffect', 'unsure' or
            'break'.  Defaults to 'break'.
        cli_change_type: str, optional
            How to resolve cli change automatically. Must be one of 'noeffect', 'unsure' or 'break'.
            Defaults to 'break'.
        config_change_type: str, optional
            How to resolve config change automatically. Must be one of 'noeffect', 'unsure' or
            'break'.  Defaults to 'break'.
    executor: `orion.executor.base.BaseExecutor`, optional
        Executor to run the experiment

    Raises
    ------
    :class:`orion.core.utils.exceptions.NoConfigurationError`
        The experiment is not in database and no space is provided by the user.
    :class:`orion.core.utils.exceptions.RaceCondition`
        There was a race condition during branching and new version cannot be inferred because of
        that. Single race conditions are normally handled seamlessly. If this error gets raised, it
        means that different modifications occurred during each race condition resolution. This is
        likely due to quick code change during experiment creation. Make sure your script is not
        generating files within your code repository.
    :class:`orion.core.utils.exceptions.BranchingEvent`
        The configuration is different than the corresponding one in DB and the branching cannot be
        solved automatically. This usually happens if the version=x is specified but the experiment
        ``(name, x)`` already has a child ``(name, x+1)``. If you really need to branch from version
        ``x``, give it a new name to branch to with ``branching={'branch_to': <new_name>}``.
    `NotImplementedError`
        If the algorithm or storage specified is not properly installed.

    """
    if max_idle_time:
        log.warning(
            "max_idle_time is deprecated. Use experiment.workon(idle_timeout) instead."
        )

    builder = experiment_builder.ExperimentBuilder(storage, debug=debug)

    try:
        experiment = builder.build(
            name,
            version=version,
            space=space,
            algorithm=algorithm,
            algorithms=algorithms,
            max_trials=max_trials,
            max_broken=max_broken,
            branching=branching,
            working_dir=working_dir,
            knowledge_base=knowledge_base,
        )
    except RaceCondition:
        # Try again, but if it fails again, raise. Race conditions due to version increment should
        # only occur once in a short window of time unless code version is changing at a crazy pace.
        try:
            experiment = builder.build(
                name,
                version=version,
                space=space,
                algorithm=algorithm,
                algorithms=algorithms,
                strategy=strategy,
                max_trials=max_trials,
                max_broken=max_broken,
                branching=branching,
                working_dir=working_dir,
                knowledge_base=knowledge_base,
            )
        except RaceCondition as e:
            raise RaceCondition(
                "There was a race condition during branching and new version cannot be inferred "
                "because of that. Single race conditions are normally handled seamlessly. If this "
                "error gets raised, it means that different modifications occurred during each "
                "race condition resolution. This is likely due to quick code change during "
                "experiment creation. Make sure your script is not generating files within your "
                "code repository."
            ) from e
    return ExperimentClient(experiment, executor, heartbeat)


def get_experiment(name, version=None, mode="r", storage=None):
    """
    Retrieve an existing experiment as :class:`orion.client.experiment.ExperimentClient`.

    Parameters
    ----------
    name: str
        The name of the experiment.
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.
    mode: str, optional
        The access rights of the experiment on the database.
        'r': read access only
        'w': can read and write to database
        Default is 'r'
    storage: dict, optional
        Configuration of the storage backend.

    Returns
    -------
    An instance of :class:`orion.client.experiment.ExperimentClient` representing the experiment.

    Raises
    ------
    `orion.core.utils.exceptions.NoConfigurationError`
        The experiment is not in the database provided by the user.
    """
    assert mode in set("rw")

    experiment = experiment_builder.load(name, version, mode, storage=storage)
    return ExperimentClient(experiment)


def workon(
    function: Callable,
    space: dict,
    name: str = "loop",
    algorithm: type[BaseAlgorithm] | str | dict | None = None,
    algorithms: type[BaseAlgorithm] | str | dict | None = None,
    max_trials: int | None = None,
    max_broken: int | None = None,
    knowledge_base: KnowledgeBase | None = None,
):
    """Optimize a function over a given search space

    This will create a new experiment with an in-memory storage and optimize the given function
    until `max_trials` is reached or the `algorithm` is done
    (some algorithm like random search are never done).

    For information on how to fetch results, see
    :py:class:`orion.client.experiment.ExperimentClient`.

    .. note::

        Each call to this function will create a separate in-memory storage.

    Parameters
    ----------
    name: str
        Name of the experiment
    version: int, optional
        Version of the experiment. Defaults to last existing version for a given `name`
        or 1 for new experiment.
    space: dict, optional
        Optimization space of the algorithm. Should have the form `dict(name='<prior>(args)')`.
    algorithm: str or dict, optional
        Algorithm used for optimization.
    max_trials: int, optional
        Maximum number or trials before the experiment is considered done.
    max_broken: int, optional
        Number of broken trials for the experiment to be considered broken.

    Raises
    ------
    `NotImplementedError`
        If the algorithm specified is not properly installed.

    """
    experiment = experiment_builder.build(
        name,
        version=1,
        space=space,
        algorithm=algorithm,
        algorithms=algorithms,
        max_trials=max_trials,
        max_broken=max_broken,
        storage={"type": "legacy", "database": {"type": "EphemeralDB"}},
        knowledge_base=knowledge_base,
    )

    experiment_client = ExperimentClient(experiment)

    with experiment_client.tmp_executor("singleexecutor", n_workers=1):
        experiment_client.workon(function, n_workers=1, max_trials=max_trials)

    return experiment_client
