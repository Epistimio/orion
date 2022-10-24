"""
Asynchronous Successive Halving Algorithm
=========================================
"""
from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Sequence

import numpy
import numpy as np

from orion.algo.hyperband import (
    BudgetTuple,
    Hyperband,
    HyperbandBracket,
    display_budgets,
)
from orion.algo.space import Space
from orion.core.worker.trial import Trial

logger = logging.getLogger(__name__)


REGISTRATION_ERROR = """
Bad fidelity level {fidelity}. Should be in {budgets}.
Params: {params}
"""

SPACE_ERROR = """
ASHA can only be used if there is one fidelity dimension.
For more information on the configuration and usage of ASHA, see
https://orion.readthedocs.io/en/develop/user/algorithms.html#asha
"""

BUDGET_ERROR = """
Cannot build budgets below max_resources;
(max: {}) - (min: {}) > (num_rungs: {})
"""


def compute_budgets(
    min_resources: float,
    max_resources: float,
    reduction_factor: int,
    num_rungs: int,
    num_brackets: int,
):
    """Compute the budgets used for ASHA"""

    # Tracks state for new trial add
    if num_brackets > num_rungs:
        logger.warning(
            "The input num_brackets %i is larger than the number of rungs %i, "
            "set num_brackets as %i",
            num_brackets,
            num_rungs,
            num_rungs,
        )
        num_brackets = num_rungs

    budgets = numpy.logspace(
        numpy.log(min_resources) / numpy.log(reduction_factor),
        numpy.log(max_resources) / numpy.log(reduction_factor),
        num_rungs,
        base=reduction_factor,
    )
    budgets = (budgets + 0.5).astype(int)

    for i in range(num_rungs - 1):
        if budgets[i] >= budgets[i + 1]:
            budgets[i + 1] = budgets[i] + 1

    if budgets[-1] > max_resources:
        raise ValueError(BUDGET_ERROR.format(min_resources, max_resources, num_rungs))

    resources = [budgets[bracket_index:] for bracket_index in range(num_brackets)]
    budgets_lists: list[list[BudgetTuple]] = []
    budgets_tab: dict[int, list[BudgetTuple]] = defaultdict(list)
    for bracket_ressources in resources:
        bracket_budgets: list[BudgetTuple] = []
        for i, min_ressources in enumerate(bracket_ressources[::-1]):
            budget = BudgetTuple(reduction_factor**i, min_ressources)
            bracket_budgets.append(budget)
            budgets_tab[len(bracket_ressources) - i - 1].append(budget)
        budgets_lists.append(bracket_budgets[::-1])

    display_budgets(budgets_tab, max_resources, reduction_factor)

    return list(budgets_lists)


class ASHA(Hyperband):
    """Asynchronous Successive Halving Algorithm

    `A simple and robust hyperparameter tuning algorithm with solid theoretical underpinnings
    that exploits parallelism and aggressive early-stopping.`

    For more information on the algorithm, see original paper at https://arxiv.org/abs/1810.05934.

    Li, Liam, et al. "Massively parallel hyperparameter tuning."
    arXiv preprint arXiv:1810.05934 (2018)

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    num_rungs: int, optional
        Number of rungs for the largest bracket. If not defined, it will be equal to ``(base + 1)``
        of the fidelity dimension. In the original paper,
        ``num_rungs == log(fidelity.high/fidelity.low) / log(fidelity.base) + 1``.
        Default: ``log(fidelity.high/fidelity.low) / log(fidelity.base) + 1``
    num_brackets: int
        Using a grace period that is too small may bias ASHA too strongly towards
        fast converging trials that do not lead to best results at convergence (stagglers). To
        overcome this, you can increase the number of brackets, which increases the amount of
        resource required for optimisation but decreases the bias towards stragglers.
        Default: 1
    repetitions: int
        Number of execution of ASHA. Default is numpy.inf which means to
        run ASHA until no new trials can be suggested.

    """

    def __init__(
        self,
        space: Space,
        seed: int | Sequence[int] | None = None,
        num_rungs: int | None = None,
        num_brackets: int = 1,
        repetitions: int | float = numpy.inf,
    ):
        super().__init__(space, seed=seed, repetitions=repetitions)

        self.num_rungs = num_rungs
        self.num_brackets = num_brackets

        self._param_names += ["num_rungs", "num_brackets"]

        if self.reduction_factor < 2:
            raise AttributeError("Reduction factor for ASHA needs to be at least 2.")

        if num_rungs is None:
            num_rungs = int(
                numpy.log(self.max_resources / self.min_resources)
                / numpy.log(self.reduction_factor)
                + 1
            )

        self.num_rungs = num_rungs

        self.budgets = compute_budgets(
            self.min_resources,
            self.max_resources,
            self.reduction_factor,
            self.num_rungs,
            self.num_brackets,
        )

        self.brackets = self.create_brackets()

        self.seed_rng(seed)

    def compute_bracket_idx(self, num: int) -> np.ndarray:
        def assign_resources(
            n: int, remainings: np.ndarray, totals: np.ndarray
        ) -> np.ndarray:
            if n == 0 or remainings.sum() == 0:
                return remainings

            ratios = remainings / totals
            fractions = numpy.nan_to_num(ratios / ratios.sum(), nan=0)
            index = numpy.argmax(fractions)
            remainings[index] -= 1
            n -= 1
            if n > 0:
                return assign_resources(n, remainings, totals)

            return remainings

        assert self.brackets is not None
        remainings = numpy.asarray(
            [bracket.remainings for bracket in self.brackets], dtype=np.int64
        )
        totals = numpy.asarray(
            [bracket.rungs[0]["n_trials"] for bracket in self.brackets], dtype=np.int64
        )
        remainings_after = copy.deepcopy(remainings)
        assign_resources(num, remainings_after, totals)
        allocations = remainings - remainings_after

        return allocations

    def sample(self, num: int) -> list[Trial]:
        samples = []
        bracket_nums = self.compute_bracket_idx(num)
        for idx, bracket_num in enumerate(bracket_nums):
            assert self.brackets is not None
            bracket = self.brackets[idx]
            bracket_samples = self.sample_from_bracket(bracket, bracket_num)
            self.register_samples(bracket, bracket_samples)
            samples += bracket_samples

        return samples

    def suggest(self, num: int) -> list[Trial]:
        return super().suggest(num)

    def create_bracket(self, budgets: list[BudgetTuple], iteration: int) -> ASHABracket:
        return ASHABracket(self, budgets, iteration)


class ASHABracket(HyperbandBracket[ASHA]):
    """Bracket of rungs for the algorithm ASHA.

    Parameters
    ----------
    asha: `ASHA` algorithm
        The asha algorithm object which this bracket will be part of.
    budgets: list of tuple
        Each tuple gives the (n_trials, resource_budget) for the respective rung.
    repetition_id: int
        The id of hyperband execution this bracket belongs to

    """

    def __init__(
        self,
        owner: ASHA,
        budgets: list[BudgetTuple],
        repetition_id: int,
    ):
        super().__init__(owner=owner, budgets=budgets, repetition_id=repetition_id)

    def get_candidates(self, rung_id: int) -> list[Trial]:
        """Get a candidate for promotion

        Raises
        ------
        TypeError
            If get_candidates is called before the entire rung is completed.
        """
        rung_results = self.rungs[rung_id]["results"]
        next_rung = self.rungs[rung_id + 1]["results"]

        rung = list(
            sorted(
                (
                    (objective, trial)
                    for objective, trial in rung_results.values()
                    if objective is not None
                ),
                key=lambda item: item[0],
            )
        )
        k = len(rung) // self.owner.reduction_factor
        k = min(k, len(rung))

        candidates: list[Trial] = []
        for i in range(k):
            trial = rung[i][1]
            _id = self.owner.get_id(trial, ignore_fidelity=True)
            if _id not in next_rung:
                candidates.append(trial)

        return candidates

    @property
    def is_filled(self) -> bool:
        """ASHA's first rung can always sample new trials"""
        return False

    def is_ready(self, rung_id: int | None = None) -> bool:
        """ASHA's always ready for promotions"""
        return True

    def promote(self, num: int) -> list[Trial]:
        """Promote the first candidate that is found and return it

        The rungs are iterated over in reversed order, so that high rungs
        are prioritised for promotions. When a candidate is promoted, the loop is broken and
        the method returns the promoted trial.

        .. note ::

            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.

        """
        if num < 1 or self.is_done:
            return []

        candidates = []
        for rung_id in range(len(self.rungs) - 2, -1, -1):
            for candidate in self.get_candidates(rung_id):
                # pylint: disable=logging-format-interpolation
                logger.debug(
                    f"Promoting {candidate} from rung {rung_id} with fidelity "
                    f"{candidate.params[self.owner.fidelity_index]} to "
                    f"rung {rung_id + 1} with fidelity {self.rungs[rung_id + 1]['resources']}"
                )

                candidate = candidate.branch(
                    status="new",
                    params={
                        self.owner.fidelity_index: self.rungs[rung_id + 1]["resources"]
                    },
                )

                if not self.owner.has_suggested(candidate):
                    candidates.append(candidate)

                if len(candidates) >= num:
                    return candidates

        return candidates
