"""
MOFA transformer stage module
"""

import numpy as np
import pandas as pd
from skopt.space import Space as SkSpace

from orion.core.utils.flatten import flatten


def fix_shape_intervals(intervals):
    """Fix issue for intervals of dims with shape
    (https://github.com/Epistimio/orion/issues/800)
    """
    for i, interval in enumerate(intervals):
        intervals[i] = tuple(map(float, interval))

    return intervals


class Transformer:
    """
    Transformer stage of MOFA

    Parameters
    ----------
    roi_space: `orion.algo.space.Space`
        Parameter region-of-interest as orion.algo.space.Space instance
    n_levels: int
        Number of levels
    """

    def __init__(self, roi_space, n_levels):
        self.n_levels = n_levels
        self.space = roi_space
        self.sk_space = SkSpace(fix_shape_intervals(roi_space.interval()))
        self.sk_space.set_transformer("normalize")

    def generate_olh_perf_table(self, trials):
        """
        Build an orthogonal Latin hypercube (OLH) performance table from trial parameters

        Parameters
        ----------
        trials: list of orion.core.worker.trial.Trial objects
            Completed trials
        """
        # TODO: deal with categoricals

        # Put trial params into list
        olh_param_table = []
        olh_objective_table = []
        for trial in trials:
            if trial.status != "completed":
                continue
            # Take subset in self.space only
            trial_params = flatten(trial.params)
            param_vals = [trial_params[key] for key in self.space]
            olh_param_table.append(param_vals)
            olh_objective_table.append(trial.objective.value)

        # Normalize
        olh_param_table = np.clip(
            np.array(olh_param_table),
            a_min=[bound[0] for bound in self.sk_space.bounds],
            a_max=[bound[1] for bound in self.sk_space.bounds],
        )
        olh_param_table = self.sk_space.transform(olh_param_table)
        table = np.hstack([olh_param_table, np.array(olh_objective_table)[:, None]])

        return pd.DataFrame(table, columns=list(self.space.keys()) + ["objective"])

    @staticmethod
    def _collapse_levels(olh_perf_table, n_levels):
        """
        Collapses the levels of an orthagonal Latin hypercube parameter table

        Parameters
        ----------
        olh_perf_table: `numpy.array`
            array of normalized trial parameters as floats in range [0., 1.]
        """
        oa_table = np.ceil(olh_perf_table.iloc[:, :-1] * (n_levels)).astype(int)
        oa_table[oa_table.iloc[:, :] == 0] = 1
        oa_table["objective"] = olh_perf_table["objective"]
        return oa_table

    def generate_oa_table(self, trials):
        """
        Generates the orthogonal array performance table

        Parameters
        ----------
        trials: list of `orion.core.worker.trial.Trial` objects
            Completed trials

        Returns
        -------
        A tuple of two numpy arrays: (OA table of parameters, OA array of trial objective values)
        """
        olh_perf_table = self.generate_olh_perf_table(trials)
        return self._collapse_levels(olh_perf_table, self.n_levels)
