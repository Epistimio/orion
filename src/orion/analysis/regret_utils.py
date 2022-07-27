"""
Provide tools to calculate regret
=================================
"""
import numpy
import pandas as pd


def regret(trials, names=("best", "best_id")):
    """
    Calculates the regret for a collection of :class:`orion.core.worker.trial.Trial`. The regret is
    calculated sequentially from the order of the collection.

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
    if len(names) != 2:
        raise ValueError(
            f"`names` requires a tuple with 2 elements. {len(names)} provided."
        )

    df = pd.DataFrame(trials, copy=True)
    if df.empty:
        return df

    regrets_idx = get_regrets_idx(df["objective"])
    df[names[0]] = df["objective"].to_numpy()[list(regrets_idx)]
    df[names[1]] = df["id"].to_numpy()[regrets_idx]

    return df


def get_regrets_idx(objectives):
    """Return the indices corresponding to the cumulative minimum"""
    minima = numpy.minimum.accumulate(objectives)
    diff = numpy.diff(minima)
    jumps = numpy.arange(len(objectives))
    jumps[1:] *= diff != 0
    jumps = numpy.maximum.accumulate(jumps)
    return jumps
