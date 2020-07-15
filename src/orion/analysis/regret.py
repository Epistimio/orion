# -*- coding: utf-8 -*-
"""
:mod:`orion.analysis.regret` -- Provide tools to calculate regret
=================================================================

.. module:: orion.analysis.regret
   :platform: Unix
   :synopsis: Provide tools to calculate regret
"""
import pandas as pd


def regret(trials, names=('best', 'best_id')):
    """
    Calculates the regret for a collection of :class:`Trial`. The regret is calculated sequentially
    from the order of the collection.

    Parameters
    ----------
    trials: DataFrame or dict
        A dataframe of trials containing, at least, the columns 'objective' and 'id'. Or a dict
        equivalent.
    names:
        A tuple containing the names of the columns. Default is ('best', 'best-id').

    Returns
    -------
    A copy of the original dataframe with two new columns containing respectively the best value
    so far and its trial id.
    """

    df = pd.DataFrame(trials, copy=True)
    if df.empty:
        return df

    df[names[0]] = df['objective'].cummin()
    df[names[1]] = __get_best_ids(df, names[0])
    return df


def __get_best_ids(dataframe, best_name):
    """Links the cumulative best objectives with their respective ids"""
    best_id = None
    result = []

    for i, id in enumerate(dataframe.id):
        if dataframe.objective[i] == dataframe[best_name][i]:
            best_id = id
        result.append(best_id)

    return result
