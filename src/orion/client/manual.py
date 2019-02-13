# -*- coding: utf-8 -*-
"""
:mod:`orion.client.manual` -- Routines for manual interaction with database
=============================================================================

.. module:: manual
   :platform: Unix
   :synopsis: Provides a simple interface to log new trials into the database
      and link them with a particular existing experiment.

"""
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.utils import format_trials


def insert_trials(experiment_name, points, cmdconfig=None, raise_exc=True):
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
    cmdconfig = cmdconfig if cmdconfig else {}
    cmdconfig['name'] = experiment_name

    experiment_view = ExperimentBuilder().build_view_from({'config': cmdconfig})

    valid_points = []

    print(experiment_view.space)

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

    for new_trial in new_trials:
        ExperimentBuilder().build_from(experiment_view.configuration).register_trial(new_trial)
