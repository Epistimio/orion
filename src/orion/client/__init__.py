# -*- coding: utf-8 -*-
"""
:mod:`orion.client` -- Helper function for returning results from script
==========================================================================

.. module:: client
   :platform: Unix
   :synopsis: Provides functions for communicating with `orion.core`.

"""
import os

from orion.client.experiment import ExperimentClient
import orion.core.io.experiment_builder as experiment_builder
from orion.core.utils.exceptions import RaceCondition
from orion.core.worker.producer import Producer


IS_ORION_ON = False
_HAS_REPORTED_RESULTS = False
RESULTS_FILENAME = os.getenv('ORION_RESULTS_PATH', None)
if RESULTS_FILENAME and os.path.isfile(RESULTS_FILENAME):
    import json
    IS_ORION_ON = True

if RESULTS_FILENAME and not IS_ORION_ON:
    raise RuntimeWarning("Results file path provided in environmental variable "
                         "does not correspond to an existing file.")


def report_results(data):
    """Facilitate the reporting of results for a user's script acting as a
    black-box computation.

    :param data: A dictionary containing experimental results

    .. note:: To be called only once in order to report a final evaluation
       of a particular trial.

    .. note:: In case that user's script is not running in a orion's context,
       this function will act as a Python `print` function.

    .. note:: For your own good, this can be called **only once**.

    """
    global _HAS_REPORTED_RESULTS  # pylint:disable=global-statement
    if _HAS_REPORTED_RESULTS:
        raise RuntimeWarning("Has already reported evaluation results once.")
    if IS_ORION_ON:
        with open(RESULTS_FILENAME, 'w') as results_file:
            json.dump(data, results_file)
    else:
        print(data)
    _HAS_REPORTED_RESULTS = True


# pylint: disable=too-many-arguments
def create_experiment(name, version=None, space=None, algorithms=None,
                      strategy=None, max_trials=None, storage=None, branching=None,
                      working_dir=None):
    """Create an experiment

    There is 2 main scenarios

    1) The experiment is new

    ``name`` and ``space`` arguments are required, otherwise ``NoConfigurationError`` will be
    raised.

    All other arguments (``algorithms``, ``strategy``, ``max_trials``, ``storage``, ``branching``
    and ``working_dir``) will be replaced by system's defaults if ommited. The system's defaults can
    also be overriden in global configuration file as described for the database in
    :ref:`Database Configuration`. We do not recommand overriding the algorithm configuration using
    system's default, but overriding the storage configuration can be very convenient if the same
    storage is used for all your experiments.

    2) The experiment exist in the database.

    We can break down this scenario in two sub-scenarios for clarity.

    2.1) Only experiment name is given.

    The configuration will be fetched from database.

    2.2) Some other arguments than the name are given.

    The configuration will be fetched from database and given arguments will override them.
    ``max_trials`` may be overwritten in DB, but any other changes will lead to a branching. Instead
    of creating the experiment ``(name, version)``, it will create a new experiment
    ``(name, version+1)`` which will have the same configuration than ``(name, version)`` except for
    the differing arguments given by user. This new experiment will have access to trials of
    ``(name, version)``, adapted according to the differences between ``version`` and ``version+1``.
    A previous version can be accessed by specifying the ``version`` argument.

    Causes of experiment branching are:
    - Change of search space
        - New dimension
        - Different prior
        - Missing dimension
    - Change of algorithm
    - Change of strategy (Not implemented yet)
    - Change of code version (Only supported by commandline API for now)

    Parameters
    ----------
    name: str
        Name of the experiment
    version: int, optional
        Version of the experiment. Defaults to last existing version for a given ``name``
        or 1 for new experiment.
    space: dict, optional
        Optimization space of the algorithm. Should have the form ``dict(name='<prior>(args)')``.
    algorithms: str or dict, optional
        Algorithm used for optimization.
    strategy: str or dict, optional
        Parallel strategy to use to parallelize the algorithm.
    max_trials: int, optional
        Maximum number or trials before the experiment is considered done.
    storage: dict, optional
        Configuration of the storage backend.
    working_dir: str, optional
        Working directory created for the experiment inside which a unique folder will be created
        for each trial. Defaults to a temporary directory that is deleted at end of execution.
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
        code_change_type: str, optional
            How to resolve code change automatically. Must be one of 'noeffect', 'unsure' or
            'break'.  Defaults to 'break'.
        cli_change_type: str, optional
            How to resolve cli change automatically. Must be one of 'noeffect', 'unsure' or 'break'.
            Defaults to 'break'.
        config_change_type: str, optional
            How to resolve config change automatically. Must be one of 'noeffect', 'unsure' or
            'break'.  Defaults to 'break'.

    Raises
    ------
    `orion.core.utils.SingletonAlreadyInstantiatedError`
        If the storage is already instantiated and given configuration is different.
        Storage is a singleton, you may only use one instance per process.
    `orion.core.utils.exceptions.NoConfigurationError`
        The experiment is not in database and no space is provided by the user.
    `orion.core.utils.exceptions.RaceCondition`
        There was a race condition during branching and new version cannot be infered because of
        that. Single race conditions are normally handled seemlessly. If this error gets raised, it
        means that different modifications occured during each race condition resolution. This is
        likely due to quick code change during experiment creation. Make sure your script is not
        generating files within your code repository.
    `ValueError`
        The configuration is different than the corresponding one in DB and the branching cannot be
        solved automatically. This usually happens if the version=x is specified but the experiment
        ``(name, x)`` already has a child ``(name, x+1)``. If you really need to branch from version
        ``x``, give it a new name to branch to with ``branching={'branch_to': <new_name>}``.
    `NotImplementedError`
        If the algorithm, storage or strategy specified is not properly installed.

    """
    experiment_builder.setup_storage(storage=storage)

    try:
        experiment = experiment_builder.build(
            name, version=version, space=space, algorithms=algorithms,
            strategy=strategy, max_trials=max_trials, branching=branching,
            working_dir=working_dir)
    except RaceCondition:
        # Try again, but if it fails again, raise. Race conditions due to version increment should
        # only occur once in a short window of time unless code version is changing at a crazy pace.
        try:
            experiment = experiment_builder.build(
                name, version=version, space=space, algorithms=algorithms,
                strategy=strategy, max_trials=max_trials, branching=branching,
                working_dir=working_dir)
        except RaceCondition as e:
            raise RaceCondition(
                "There was a race condition during branching and new version cannot be infered "
                "because of that. Single race conditions are normally handled seemlessly. If this "
                "error gets raised, it means that different modifications occured during each race "
                "condition resolution. This is likely due to quick code change during experiment "
                "creation. Make sure your script is not generating files within your code "
                "repository.") from e

    producer = Producer(experiment)

    return ExperimentClient(experiment, producer)
