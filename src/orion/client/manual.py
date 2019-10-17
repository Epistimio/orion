# -*- coding: utf-8 -*-
"""
:mod:`orion.client.manual` -- Routines for manual interaction with database
=============================================================================

.. module:: manual
   :platform: Unix
   :synopsis: Provides a simple interface to log new trials into the database
      and link them with a particular existing experiment.

"""
import orion.core.io.experiment_builder as experiment_builder
from orion.core.utils import format_trials


def insert_trials(experiment_name, points, raise_exc=True):
    """Insert sets of parameters manually, defined in `points`, as new trials
    for the experiment name, `experiment_name`.

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
    experiment_view = experiment_builder.build_view(name=experiment_name)

    valid_points = []

    for point in points:
        try:
            assert point in experiment_view.space
            valid_points.append(point)
        except AssertionError:
            if raise_exc:
                raise

    if not valid_points:
        return

    new_trials = list(
        map(lambda data: format_trials.tuple_to_trial(data, experiment_view.space),
            valid_points))

    experiment = experiment_builder.build(name=experiment_view.name,
                                          version=experiment_view.version)
    for new_trial in new_trials:
        experiment.register_trial(new_trial)
