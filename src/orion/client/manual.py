# -*- coding: utf-8 -*-
"""
Routines for manual interaction with database
===============================================

Provides a simple interface to log new trials into the database and link them with a particular
existing experiment.

"""
import logging

from orion.client import create_experiment
from orion.core.utils import format_trials

log = logging.getLogger(__name__)


def insert_trials(experiment_name, points, raise_exc=True):
    """Insert sets of parameters manually, defined in `points`, as new trials
    for the experiment name, `experiment_name`.

    .. warning::

        This function is deprecated and will be removed in 0.3.0.
        You should use ExperimentClient.insert() instead.

    :param experiment_name: Name of the experiment which the new trials are
       going to be associated with
    :param points: list of tuples in agreement with experiment's parameter space
    :param raise_exc: whether an inappropriate tuple of parameters will raise
       an exception or it will be ignored

    .. note:: If `raise_exc` is True, no set of parameters will be inserted. If
       it is False, only the valid ones will be inserted; the rest will be ignored.

    .. note:: This cannot be used to prepopulate a future experiment. So,
       an experiment with `experiment_name` should already be configured in
       the database.

    """
    log.warning(
        "insert_trials() is deprecated and will be removed in 0.3.0. "
        "You should use ExperimentClient.insert() instead."
    )
    experiment = create_experiment(experiment_name)

    valid_points = []

    for point in points:
        try:
            assert point in experiment.space
            valid_points.append(point)
        except AssertionError:
            if raise_exc:
                raise

    if not valid_points:
        return

    new_trials = list(
        map(
            lambda data: format_trials.tuple_to_trial(data, experiment.space),
            valid_points,
        )
    )

    for new_trial in new_trials:
        experiment.insert(new_trial.params)
