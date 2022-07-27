"""
:mod:`orion.algo.mofa.mofa` -- MOFA
============================================

MOdular FActorial Design (MOFA)

"""
from __future__ import annotations

# pylint: disable=attribute-defined-outside-init,invalid-name
import copy
import logging
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
import scipy
from packaging import version

from orion.algo.base import BaseAlgorithm
from orion.algo.mofa import sampler
from orion.algo.mofa.transformer import Transformer
from orion.algo.space import Real, Space
from orion.core.utils.format_trials import dict_to_trial
from orion.core.worker.trial import Trial

logger = logging.getLogger(__name__)


class MOFA(BaseAlgorithm):
    """
    MOdular FActorial Design (MOFA).

    For more information on the algorithm, see original paper: MOFA: Modular
    Factorial Design for Hyperparameter Optimization https://arxiv.org/abs/2011.09545

    Xiong, Bo, Yimin Huang, Hanrong Ye, Steffen Staab, and Zhenguo Li.
    "MOFA: Modular Factorial Design for Hyperparameter Optimization."
    arXiv preprint arXiv:2011.09545 (2020).

    Notes
    -----
    Default values for the index, n_levels, and strength (t) parameter are set
    to the empirically obtained optimal values described in section 5.2 of the paper.

    The number of trials N for a single MOFA iteration is set to ``N = index * n_levels^t``.
    The ``--exp-max-trials`` should be a multiple of N.

    MOFA requires Python v3.8 or greater and scipy v1.8 or greater.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    index: int, optional
        This is the lambda parameter in the paper.
        Default: ``1``
    n_levels: int, optional
        Number of levels in the orthogonal Latin hypercube (OLH) table. Should be set to a
        prime number. This is the l parameter in the paper.
        Default: ``5``
    strength: int, optional
        Strength parameter. This is the t parameter in the paper.
        Default: ``2``
    threshold: float, optional
        The threshold to determine is a dimension was explored enough and can be fixed.
        Default: 0.1

    """

    requires_type = "real"
    requires_dist = "linear"
    requires_shape = "flattened"

    def __init__(
        self,
        space: Space,
        seed: int | Sequence[int] | None = None,
        index: int = 1,
        n_levels: int = 5,
        strength: int = 2,
        threshold: float = 0.1,
    ):
        if version.parse(scipy.__version__) < version.parse("1.8"):
            raise RuntimeError("MOFA algorithm requires scipy version >= 1.8.")

        index = int(index)
        n_levels = int(n_levels)
        threshold = float(threshold)
        if index < 1:
            raise ValueError(f"index must be >= 1! (currently: {index})!\n")
        if n_levels not in {2, 3, 5, 7, 11, 13, 17, 19}:
            raise ValueError(
                f"n_levels must be a small prime number! (currently: {n_levels})!\n"
                f"The implementation additionally restricts it to <= 19."
            )
        if strength not in {1, 2}:
            raise ValueError(f"strength must be 1 or 2 (currently: {strength})!\n")
        if threshold <= 0 or threshold >= 1:
            raise ValueError(
                f"threshold must be strictly between 0 and 1! (currently: {threshold})!\n"
            )

        super().__init__(
            space,
            seed=seed,
            index=index,
            n_levels=n_levels,
            strength=strength,
            threshold=threshold,
        )

        self.cur_n_levels = self.n_levels
        self.roi_space = None
        self.completed_trials = []
        self.duplicates = defaultdict(list)
        self.converged = False
        self.n_trials = None

        # Initialize a dictionary for the frozen values of each hyperparameter
        self.frozen_param_values = {}

        # Generate the initial set of trials
        self.roi_space = self.space
        self.current_trials_params = self._generate_trials_unfrozen_params(
            self.roi_space
        )

    def _generate_trials_unfrozen_params(self, roi_space: Space) -> list[dict]:
        olh_samples, self.cur_n_levels = sampler.generate_olh_samples(
            roi_space, self.n_levels, self.strength, self.index, self.rng
        )
        self.n_trials = len(olh_samples)
        olh_trials = sampler.generate_trials(olh_samples, roi_space)
        return olh_trials

    def seed_rng(self, seed: int) -> None:
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int
            Integer seed for the random number generator.

        """
        self.rng = np.random.RandomState(seed)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super().state_dict
        state_dict["rng_state"] = self.rng.get_state()
        state_dict["completed_trials"] = copy.deepcopy(self.completed_trials)
        state_dict["duplicates"] = copy.deepcopy(self.duplicates)
        state_dict["converged"] = self.converged
        state_dict["roi_space"] = self.roi_space
        state_dict["frozen_param_values"] = copy.deepcopy(self.frozen_param_values)
        if hasattr(self, "current_trials_params"):
            state_dict["current_trials_params"] = copy.deepcopy(
                self.current_trials_params
            )
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        Parameters
        ----------
        state_dict: dict
            Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)
        self.seed_rng(0)
        self.rng.set_state(state_dict["rng_state"])
        self.completed_trials = copy.deepcopy(state_dict["completed_trials"])
        self.duplicates = copy.deepcopy(state_dict["duplicates"])
        self.converged = state_dict["converged"]
        self.roi_space = copy.deepcopy(state_dict["roi_space"])
        self.frozen_param_values = copy.deepcopy(state_dict["frozen_param_values"])
        if "current_trials_params" in state_dict:
            self.current_trials_params = copy.deepcopy(
                state_dict["current_trials_params"]
            )
        elif hasattr(self, "current_trials_params"):
            self.current_trials_params = []

    def suggest(self, num: int) -> list[Trial]:
        """Suggest a number of new sets of parameters.

        Draws points from a prepared set of samples from an orthonal Latin hypercube.

        Parameters
        ----------
        num: int, optional
            Number of trials to suggest. The algorithm may return less than the number of trials
            requested.

        Returns
        -------
        list of trials
            A list of trials representing values suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return an empty list.


        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """

        if self.n_trials is None or len(self.completed_trials) >= self.n_trials:
            self._prepare_next_iteration()

        trials = []
        while (
            len(trials) < num
            and len(self.current_trials_params) > 0
            and not self.is_done
        ):
            trial_params = self.current_trials_params.pop()
            trial_params.update(self.frozen_param_values)
            trial = dict_to_trial(trial_params, self.space)
            if self.has_observed(trial):
                similar_trial = self.registry.get_existing(trial)
                trial.results = copy.deepcopy(similar_trial.results)
                trial.status = similar_trial.status
                self.completed_trials.append(trial)
            elif self.has_suggested(trial):
                self.duplicates[self.get_id(trial)].append(trial)
            else:
                self.register(trial)
                trials.append(trial)

        if len(trials) == 0 and len(self.completed_trials) >= self.n_trials:
            self.converged = True

        return trials

    def observe(self, trials: list[Trial]) -> None:
        """Observe the `trials` new state of result.

        Collects the completed trials until all trials for the current MOFA iteration have been
        provided. Then runs the MOFA transformer, analysis and region-of-interest generation
        stages to prepare for the next iteration, or stops if all parameters have been frozen.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """

        # Collect trials until all trials have been completed. Otherwise return.
        for trial in trials:
            if not self.has_suggested(trial):
                logger.debug(
                    "Ignoring trial %s because it was not sampled by current algo.",
                    trial,
                )
                continue
            self.register(trial)

            if trial.status in ["completed", "broken"]:
                self.completed_trials.append(trial)
                for duplicate_trial in self.duplicates[self.get_id(trial)]:
                    duplicate_trial.results = copy.deepcopy(trial.results)
                    duplicate_trial.status = trial.status
                    self.completed_trials.append(duplicate_trial)

        if self.n_trials is not None and len(self.completed_trials) >= self.n_trials:
            self._prepare_next_iteration()

    def _prepare_next_iteration(self) -> None:
        # Transformer stage
        transformer = Transformer(self.roi_space, self.cur_n_levels)
        oa_table = transformer.generate_oa_table(self.completed_trials)

        # Reset the completed_trials list
        self.completed_trials.clear()
        self.duplicates.clear()

        # Analyze stage
        factorial_performance_analysis = get_factorial_performance_analysis(
            oa_table, self.roi_space, self.cur_n_levels
        )
        factorial_importance_analysis = get_factorial_importance_analysis(
            factorial_performance_analysis, self.roi_space
        )

        # Select new Region of Interest.
        self.roi_space, frozen_param_values = select_new_region_of_interest(
            factorial_importance_analysis,
            self.roi_space,
            self.threshold,
            self.cur_n_levels,
        )
        self.frozen_param_values.update(frozen_param_values)
        if (set(self.frozen_param_values.keys()) | set(self.roi_space.keys())) != set(
            self.space.keys()
        ):
            raise RuntimeError(
                "Some dimensions were lost during selection of new region of interest.\n"
                f"Frozen parameters: {sorted(self.frozen_param_values.keys())}\n"
                f"Region of interest: {sorted(self.roi_space.keys())}\n"
                f"Space: {sorted(self.space.keys())}"
            )

        if self.is_done:
            return

        # Create new set of trials
        self.current_trials_params = self._generate_trials_unfrozen_params(
            self.roi_space
        )

        return

    @property
    def is_done(self) -> bool:
        """Return True, if an algorithm holds that there can be no further improvement."""
        if len(self.frozen_param_values.items()) == len(self.space.items()):
            return True

        return self.converged or super().is_done


def get_factorial_performance_analysis(
    oa_table: pd.DataFrame, space: Space, n_levels: int
) -> pd.DataFrame:
    """Compute the factorial performance analysis"""
    levels = list(range(1, n_levels + 1))
    factorial_performance_analysis = [[level] for level in levels]
    for key in space.keys():
        marginal_means = (
            oa_table[[key, "objective"]].groupby([key]).mean().reset_index()
        )
        dim_levels = set(levels)
        for _, row in marginal_means.iterrows():
            dim_level = int(row[key])
            factorial_performance_analysis[dim_level - 1].append(row["objective"])
            dim_levels.remove(dim_level)

        for remaining_level in dim_levels:
            factorial_performance_analysis[remaining_level - 1].append(float("inf"))

    return pd.DataFrame(
        factorial_performance_analysis, columns=["level"] + list(space.keys())
    )


def get_factorial_importance_analysis(
    factorial_performance_analysis: pd.DataFrame, space: int
) -> pd.DataFrame:
    """Compute the factorial importance analysis"""
    factorial_importance_analysis = []
    total_marginal_variance = (
        np.ma.masked_invalid(
            factorial_performance_analysis[list(space.keys())].to_numpy()
        )
        .var(0)
        .sum()
    )

    for key in space.keys():
        best_level = factorial_performance_analysis[key].argmin() + 1
        importance = (
            np.ma.masked_invalid(factorial_performance_analysis[key].to_numpy()).var()
            / total_marginal_variance
        )
        factorial_importance_analysis.append((key, best_level, importance))

    return pd.DataFrame(
        factorial_importance_analysis, columns=["param", "best_level", "importance"]
    )


def select_new_region_of_interest(
    factorial_importance_analysis: pd.DataFrame,
    space: Space,
    threshold: float,
    n_levels: int,
) -> tuple[Space, dict]:
    """Select new region of interest and frozen parameter values based on factorial analysis.

    Parameters
    ----------
    factorial_importance_analysis: dict
        Marginal variance ratios on best levels of the factorial performance analysis.
        Should have format {'dim-name': <marginal variance ratio>, ...}
    space: ``orion.algo.space.Space``
        Space object representing the current region of interest.
    threshold: float
        Threshold of marginal variance ratio below which we should freeze a dimension.

    """

    frozen_param_values = {}
    new_space = Space()
    for key, dim in space.items():
        dim_importance_analysis = factorial_importance_analysis[
            factorial_importance_analysis["param"] == key
        ]
        if float(dim_importance_analysis["importance"]) < threshold:
            frozen_param_values[key] = sum(dim.interval()) / 2.0
        else:
            level = int(dim_importance_analysis["best_level"])
            low, high = dim.interval()
            intervals = (high - low) / n_levels
            new_low = low + intervals * (level - 1)
            new_space.register(Real(dim.name, "uniform", new_low, intervals))

    return new_space, frozen_param_values
