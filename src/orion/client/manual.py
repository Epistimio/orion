# -*- coding: utf-8 -*-
"""
:mod:`orion.client.manual` -- Routines for manual interaction with database
=============================================================================

.. module:: manual
   :platform: Unix
   :synopsis: Provides a simple interface to log new trials into the database
      and link them with a particular existing experiment.

"""
from orion.core.cli import resolve_config
from orion.core.io.database import Database
from orion.core.utils import (format_trials, SingletonError,)
from orion.core.worker.experiment import Experiment


def insert_trials(experiment_name, points, cmdconfig=None, raise_exc=True):
    """Insert sets of parameters manually, defined in `points`, as new trials
    for the experiment name, `experiment_name`.

    :param experiment_name: Name of the experiment which the new trials are
       going to be associated with
    :param points: list of tuples in agreement with experiment's parameter space
    :param raise_exc: whether an inappropriate tuple of parameters will raise
       an exception or it will be ignored

    .. info:: If `raise_exc` is True, no set of parameters will be inserted. If
       it is False, only the valid ones will be inserted; the rest will be ignored.

    .. note:: This cannot be used to prepopulate a future experiment. So,
       an experiment with `experiment_name` should already be configured in
       the database.

    """
    cmdconfig = cmdconfig if cmdconfig else {}
    config = resolve_config.fetch_default_options()  # Get database perhaps from default locs
    config = resolve_config.merge_env_vars(config)  # Get database perhaps from env vars

    tmpconfig = resolve_config.merge_orion_config(config, dict(),
                                                  cmdconfig, dict())

    db_opts = tmpconfig['database']
    db_type = db_opts.pop('type')
    try:
        Database(of_type=db_type, **db_opts)
    except SingletonError:
        pass

    experiment = Experiment(experiment_name)
    # Configuration is completely taken from the database
    if experiment.status is None:
        raise ValueError("No experiment named '{}' could be found.".format(experiment_name))
    experiment.configure(experiment.configuration)

    valid_points = []

    print(experiment.space)

    for point in points:
        try:
            assert point in experiment.space
            valid_points.append(point)
        except AssertionError:
            if raise_exc:
                raise

    if not valid_points:
        return

    new_trials = list(map(lambda data: format_trials.tuple_to_trial(data,
                                                                    experiment.space),
                          valid_points))
    experiment.register_trials(new_trials)
