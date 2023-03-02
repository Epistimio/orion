"""
:mod:`orion.algo.pbt.pb2
========================

"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any, ClassVar, Sequence

import numpy as np
import pandas

from orion.algo.pbt.pb2_utils import import_optional, select_config
from orion.algo.pbt.pbt import PBT
from orion.core.utils.flatten import flatten
from orion.core.utils.random_state import RandomState, control_randomness
from orion.core.worker.transformer import ReshapedSpace, TransformedSpace
from orion.core.worker.trial import Trial

logger = logging.getLogger(__name__)


class PB2(PBT):
    """Population Based Bandits

    Warning: PB2 is broken in current version v0.2.4. We are working on a fix to be released in
    v0.2.5, ETA July 2022.

    Population Based Bandits is a variant of Population Based Training using probabilistic model
    to guide the search instead of relying on purely random perturbations.
    PB2 implementation uses a time-varying Gaussian process to model the optimization curves
    during training. This implementation is based on ray-tune implementation. Oríon's version
    supports discrete and categorical dimensions, and offers better resiliency to broken
    trials by using back-tracking.

    See PBT documentation for more information on how to use PBT algorithms.

    For more information on the algorithm,
    see original paper at https://arxiv.org/abs/2002.02518.

    Parker-Holder, Jack, Vu Nguyen, and Stephen J. Roberts.
    "Provably efficient online hyperparameter optimization with population-based bandits."
    Advances in Neural Information Processing Systems 33 (2020): 17200-17211.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    population_size: int, optional
        Size of the population. No trial will be continued until there are `population_size`
        trials executed until lowest fidelity. If a trial is broken during execution at lowest
        fidelity, the algorithm will sample a new trial, keeping the population of *non-broken*
        trials at `population_size`.  For efficiency it is better to have less workers running than
        population_size. Default: 50.
    generations: int, optional
        Number of generations, from lowest fidelity to highest one. This will determine how
        many branchings occur during the execution of PBT. Default: 10
    exploit: dict or None, optional
        Configuration for a ``pbt.exploit.BaseExploit`` object that determines
        when if a trial should be exploited or not. If None, default configuration
        is a ``PipelineExploit`` with ``BacktrackExploit`` and ``TruncateExploit``.
    fork_timeout: int, optional
        Maximum amount of time in seconds that an attempt to mutate a trial should take, otherwise
        algorithm.suggest() will raise ``SuggestionTimeout``. Default: 60

    """

    requires_type: ClassVar[str | None] = "real"
    requires_dist: ClassVar[str | None] = "linear"
    requires_shape: ClassVar[str | None] = "flattened"

    def __init__(
        self,
        space,
        seed=None,
        population_size=50,
        generations=10,
        exploit=None,
        fork_timeout=60,
    ):
        import_optional.ensure()

        self.random_state: RandomState | None = None

        super().__init__(
            space,
            seed=seed,
            population_size=population_size,
            generations=generations,
            exploit=exploit,
            fork_timeout=fork_timeout,
        )

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        """
        config = copy.deepcopy(super().configuration)
        config["pb2"].pop("explore", None)
        return config

    def seed_rng(self, seed: int | Sequence[int] | None) -> None:
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int
            Integer seed for the random number generator.
        """
        super().seed_rng(seed)
        self.random_state = RandomState.seed(self.rng.randint(0, 2**32 - 1))

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict: dict[str, Any] = super().state_dict
        state_dict["random_state"] = self.random_state or RandomState.current()
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict"""
        super().set_state(state_dict)
        self.random_state = state_dict["random_state"]

    def _generate_offspring(self, trial):
        """Try to promote or fork a given trial."""

        new_trial = trial

        if not self.has_suggested(new_trial):
            raise RuntimeError(
                "Trying to fork a trial that was not registered yet. This should never happen"
            )

        attempts = 0
        start = time.perf_counter()
        while (
            self.has_suggested(new_trial)
            and time.perf_counter() - start <= self.fork_timeout
        ):
            trial_to_explore = self.exploit_func(
                self.rng,
                trial,
                self.lineages,
            )

            if trial_to_explore is None:
                return None, None
            elif trial_to_explore is trial:
                new_params = {}
                trial_to_branch = trial
                logger.debug("Promoting trial %s, parameters stay the same.", trial)
            else:
                new_params = flatten(self._explore(self.space, trial_to_explore))
                trial_to_branch = trial_to_explore
                logger.debug(
                    "Forking trial %s with new parameters %s",
                    trial_to_branch,
                    new_params,
                )

            # Set next level of fidelity
            new_params[self.fidelity_index] = self.fidelity_upgrades[
                flatten(trial_to_branch.params)[self.fidelity_index]
            ]

            new_trial = trial_to_branch.branch(params=new_params)
            assert isinstance(self.space, (TransformedSpace, ReshapedSpace))
            new_trial = self.space.transform(self.space.reverse(new_trial))

            logger.debug("Attempt %s - Creating new trial %s", attempts, new_trial)

            attempts += 1

        if (
            self.has_suggested(new_trial)
            and time.perf_counter() - start > self.fork_timeout
        ):
            trial_to_branch = None
            new_trial = None
            logger.info(
                f"Could not generate unique new parameters for trial {trial.id} in "
                f"less than {self.fork_timeout} seconds. Attempted {attempts} times."
            )

        return trial_to_branch, new_trial

    def _explore(self, space, base: Trial):
        """Generate new hyperparameters for given trial.

        Derived from PB2 explore implementation in Ray (2022/02/18):
        https://github.com/ray-project/ray/blob/master/python/ray/tune/schedulers/pb2.py#L131
        """

        base_params = flatten(base.params)

        data, current = self._get_data_and_current()
        bounds = {dim.name: dim.interval() for dim in space.values()}

        df = data.copy()

        # Group by trial ID and hyperparams.
        # Compute change in timesteps and reward.
        diff_reward = (
            df.groupby(["Trial"] + list(bounds.keys()))["Reward"]
            .mean()
            .diff()
            .reset_index(drop=True)
        )
        df["y"] = diff_reward

        df["R_before"] = df.Reward - df.y

        df = df[~df.y.isna()].reset_index(drop=True)

        # Only use the last 1k datapoints, so the GP is not too slow.
        df = df.iloc[-1000:, :].reset_index(drop=True)

        # We need this to know the T and Reward for the weights.
        if not df[df["Trial"] == self.get_id(base)].empty:
            # N ow specify the dataset for the GP.
            y_raw = np.array(df.y.values)
            # Meta data we keep -> episodes and reward.
            t_r = df[["Budget", "R_before"]]
            hparams = df[bounds.keys()]
            x_raw = pandas.concat([t_r, hparams], axis=1).values
            newpoint = (
                df[df["Trial"] == self.get_id(base)]
                .iloc[-1, :][["Budget", "R_before"]]
                .values
            )
            with control_randomness(self):
                new = select_config(
                    x_raw,
                    y_raw,
                    current,
                    newpoint,
                    bounds,
                    num_f=len(t_r.columns),
                )

            new_config = base_params.copy()
            for i, col in enumerate(hparams.columns):
                if isinstance(base_params[col], int):
                    new_config[col] = int(new[i])
                else:
                    new_config[col] = new[i]

        else:
            new_config = base_params

        return new_config

    def _get_data_and_current(self):
        """Generate data and current objects used in _explore function.

        data is a pandas DataFrame combining data from all completed trials.
        current is a numpy array with hyperparameters from uncompleted trials.
        """
        data_trials = []
        current_trials = []
        for trial in self.registry:
            if trial.status == "completed":
                data_trials.append(trial)
            else:
                current_trials.append(trial)
        data = self._trials_to_data(data_trials)
        if current_trials:
            current_array = []
            for trial in current_trials:
                trial_params = flatten(trial.params)
                current_array.append([trial_params[key] for key in self.space.keys()])
            current = np.asarray(current_array)
        else:
            current = None
        return data, current

    def _trials_to_data(self, trials):
        """Generate data frame to use in _explore method."""
        rows = []
        cols = ["Trial", "Budget"] + list(self.space.keys()) + ["Reward"]
        for trial in trials:
            trial_params = flatten(trial.params)
            values = [trial_params[key] for key in self.space.keys()]
            lst = (
                [self.get_id(trial), trial_params[self.fidelity_index]]
                + values
                + [trial.objective.value]
            )
            rows.append(lst)
        data = pandas.DataFrame(rows, columns=cols)
        data.Trial = data.Trial.astype("str")
        return data
