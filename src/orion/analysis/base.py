# -*- coding: utf-8 -*-
"""
:mod:`orion.analysis.base` -- Provide base tools for analysis of results
========================================================================

.. module:: orion.analysis.base
   :platform: Unix
   :synopsis: Provide base tools for analysis of results
"""
import numpy


def average(trials, group_by="order", key="best", return_var=False):
    """Compute the average of some trial attribute.

    By default it will compute the average objective at each time step across
    multiple experiments.

    Parameters
    ----------
    trials: DataFrame
        A dataframe of trials containing, at least, the columns 'best' and 'order'.
    group_by: str, optional
        The attribute to use to group trials for the average. By default it group trials
        by order (ex: all first trials across experiments.)
    key: str, optional
        The attribute to average. Defaults to 'best' as returned by ``orion.analysis.regret``.
    return_var: bool, optional
        If True, and a column '{key}_var' where '{key}' is the value of the argument `key`.
        Defaults to False.

    Returns
    -------
    A dataframe with columns 'order', '{key}_mean' and '{key}_var'.

    """
    if trials.empty:
        return trials

    group = trials.groupby(group_by)
    mean = group[key].mean().reset_index().rename(columns={key: f"{key}_mean"})
    if return_var:
        mean[f"{key}_var"] = group[key].var().reset_index()[key]

    return mean


def ranking(trials, group_by="order", key="best"):
    """Compute the ranking of some trial attribute.

    By default it will compute the ranking with respect to objectives at each time step across
    multiple experiments.

    Parameters
    ----------
    trials: DataFrame
        A dataframe of trials containing, at least, the columns 'best' and 'order'.
    group_by: str, optional
        The attribute to use to group trials for the ranking. By default it group trials
        by order (ex: all first trials across experiments.)
    key: str, optional
        The attribute to use for the ranking. Defaults to 'best' as returned by
        ``orion.analysis.regret``.

    Returns
    -------
    A copy of the original dataframe with a new column 'rank' for the rankings.

    """
    if trials.empty:
        return trials

    def rank(row):
        indices = row[key].argsort()
        ranks = numpy.empty_like(indices)
        ranks[indices] = numpy.arange(len(ranks))
        row["rank"] = ranks
        return row

    return trials.groupby(group_by).apply(rank)
