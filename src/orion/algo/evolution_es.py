"""
The Evolved Transformer and large-scale evolution of image classifiers
======================================================================

Implement evolution to exploit configurations with fixed resource efficiently

"""
from __future__ import annotations

import copy
import importlib
import logging
from typing import Callable, ClassVar, Sequence, TypeVar

import numpy as np

from orion.algo.hyperband import BudgetTuple, Hyperband, HyperbandBracket
from orion.algo.space import Space
from orion.core.utils import format_trials
from orion.core.worker.trial import Trial

logger = logging.getLogger(__name__)

REGISTRATION_ERROR = """
Bad fidelity level {fidelity}. Should be in {budgets}.
Params: {params}
"""

SPACE_ERROR = """
EvolutionES cannot be used if space does not contain a fidelity dimension.
"""

BUDGET_ERROR = """
Cannot build budgets below max_resources;
(max: {}) - (min: {}) > (num_rungs: {})
"""


def compute_budgets(
    min_resources: int,
    max_resources: int,
    reduction_factor: int,
    nums_population: int,
    pairs: int,
) -> list[list[BudgetTuple]]:
    """Compute the budgets used for each execution of hyperband"""
    budgets_eves = []
    if reduction_factor == 1:
        for i in range(min_resources, max_resources + 1):
            if i == min_resources:
                budgets_eves.append([BudgetTuple(nums_population, i)])
            else:
                budgets_eves[0].append(BudgetTuple(pairs * 2, i))
    else:
        num_brackets = int(np.log(max_resources) / np.log(reduction_factor))
        budgets: list[list[BudgetTuple]] = []
        budgets_tab: dict[int, list[BudgetTuple]] = {}  # just for display consideration
        for bracket_id in range(0, num_brackets + 1):
            bracket_budgets: list[BudgetTuple] = []
            num_trials = int(
                np.ceil(
                    int((num_brackets + 1) / (num_brackets - bracket_id + 1))
                    * (reduction_factor ** (num_brackets - bracket_id))
                )
            )

            min_resources = max_resources / reduction_factor ** (
                num_brackets - bracket_id
            )
            for i in range(0, num_brackets - bracket_id + 1):
                n_i = int(num_trials / reduction_factor**i)
                min_i = int(min_resources * reduction_factor**i)
                bracket_budgets.append(BudgetTuple(n_i, min_i))

                budget = BudgetTuple(n_i, min_i)
                if budgets_tab.get(i):
                    budgets_tab[i].append(budget)
                else:
                    budgets_tab[i] = [budget]

            budgets.append(bracket_budgets)

        for i in range(len(budgets[0])):
            if i == 0:
                budgets_eves.append([BudgetTuple(nums_population, budgets[0][i][1])])
            else:
                budgets_eves[0].append(BudgetTuple(pairs * 2, budgets[0][i][1]))

    return budgets_eves


BracketT = TypeVar("BracketT", bound="BracketEVES")


class EvolutionES(Hyperband[BracketT]):
    """EvolutionES formulates hyperparameter optimization as an evolution.

    For more information on the algorithm,
    see original paper at
    https://arxiv.org/pdf/1703.01041.pdf and
    https://arxiv.org/pdf/1901.11117.pdf

    Real et al. "Large-Scale Evolution of Image Classifiers"
    So et all. "The Evolved Transformer"

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    repetitions: int
        Number of execution of Hyperband. Default is numpy.inf which means to
        run Hyperband until no new trials can be suggested.
    nums_population: int
        Number of population for EvolutionES. Larger number of population often gets better
        performance but causes more computation. So there is a trade-off according to the search
        space and required budget of your problems.
        Default: 20
    mutate: str or dict or None, optional
        In the mutate part, one can define the customized mutate function with its mutate factors,
        such as multiply factor (times/divides by a multiply factor) and add factor
        (add/subtract by a multiply factor). The function must be defined by
        an importable string. If None, default
        mutate function is used: ``orion.algo.mutate_functions.default_mutate``.

    """

    requires_type: ClassVar[str | None] = None
    requires_dist: ClassVar[str | None] = None
    requires_shape: ClassVar[str | None] = "flattened"

    def __init__(
        self,
        space: Space,
        seed: int | Sequence[int] | None = None,
        repetitions: int | float = np.inf,
        nums_population: int = 20,
        mutate: str | dict | None = None,
        max_retries: int = 1000,
    ):
        self.mutate = mutate

        super().__init__(space, seed=seed, repetitions=repetitions)
        pair = nums_population // 2
        mutate_ratio = 0.3
        self.nums_population = nums_population
        self.nums_comp_pairs = pair
        self.max_retries = max_retries
        self.mutate_ratio = mutate_ratio
        self.nums_mutate_gene = (
            int((len(self.space.values()) - 1) * mutate_ratio)
            if int((len(self.space.values()) - 1) * mutate_ratio) > 0
            else 1
        )

        self.hurdles: list[float | np.ndarray] = []

        self.population: dict[int, list[int]] = {}
        for i, dim in enumerate(self.space.values()):
            if dim.type != "fidelity":
                self.population[i] = [-1] * nums_population

        self.performance: np.ndarray = np.inf * np.ones(nums_population)

        self.budgets = compute_budgets(
            self.min_resources,
            self.max_resources,
            self.reduction_factor,
            nums_population,
            pair,
        )

        self.brackets: list[BracketT] = self.create_brackets()
        self.seed_rng(seed)

    def create_bracket(
        self, bracket_budgets: list[BudgetTuple], iteration: int
    ) -> BracketT:
        return BracketEVES(self, bracket_budgets, iteration)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super().state_dict
        state_dict["population"] = copy.deepcopy(self.population)
        state_dict["performance"] = copy.deepcopy(self.performance)
        state_dict["hurdles"] = copy.deepcopy(self.hurdles)
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict"""
        self.population = state_dict["population"]
        self.performance = state_dict["performance"]
        self.hurdles = state_dict["hurdles"]
        super().set_state(state_dict)

    def _get_bracket(self, trial: Trial) -> BracketT:
        """Get the bracket of a trial during observe"""
        return self.brackets[-1]


class BracketEVES(HyperbandBracket[EvolutionES]):
    """Bracket of rungs for the algorithm Hyperband.

    Parameters
    ----------
    evolutiones: `evolutiones` algorithm
        The evolutiones algorithm object which this bracket will be part of.
    budgets: list of tuple
        Each tuple gives the (n_trials, resource_budget) for the respective rung.
    repetition_id: int
        The id of hyperband execution this bracket belongs to

    """

    def __init__(
        self, owner: EvolutionES, budgets: list[BudgetTuple], repetition_id: int
    ):
        super().__init__(owner, budgets, repetition_id)
        self.search_space_without_fidelity = []
        self._candidates: dict[int, list[Trial]] = {}

        self.mutate_attr: dict = {}
        if owner.mutate:
            if isinstance(owner.mutate, str):
                self.mutate_attr = {"function": owner.mutate}
            elif isinstance(owner.mutate, dict):
                self.mutate_attr = copy.deepcopy(owner.mutate)
            else:
                raise ValueError(f"Unsupported type for mutate: {owner.mutate}")
        function_string = self.mutate_attr.pop(
            "function", "orion.algo.mutate_functions.default_mutate"
        )
        mod_name, func_name = function_string.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        self.mutate_func: Callable = getattr(mod, func_name)

        for i, dim in enumerate(self.space.values()):
            if dim.type != "fidelity":
                self.search_space_without_fidelity.append(i)

    @property
    def space(self) -> Space:
        return self.owner.space

    @property
    def state_dict(self) -> dict:
        state_dict = super().state_dict
        state_dict["candidates"] = copy.deepcopy(self._candidates)
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        super().set_state(state_dict)
        self._candidates = state_dict["candidates"]

    def _get_teams(self, rung_id: int) -> list | tuple[dict, int, list[int], list[int]]:
        """Get the red team and blue team"""
        if self.has_rung_filled(rung_id + 1):
            return []

        rung = self.rungs[rung_id]["results"]

        population_range = (
            self.owner.nums_population
            if len(list(rung.values())) > self.owner.nums_population
            else len(list(rung.values()))
        )

        rung_trials = list(rung.values())
        for trial_index in range(population_range):
            objective, trial = rung_trials[trial_index]
            self.owner.performance[trial_index] = objective
            for ith_dim in self.search_space_without_fidelity:
                self.owner.population[ith_dim][trial_index] = trial.params[
                    self.space[ith_dim].name
                ]

        population_index = list(range(self.owner.nums_population))
        red_team = self.owner.rng.choice(
            population_index, self.owner.nums_comp_pairs, replace=False
        )
        diff_list = list(set(population_index).difference(set(red_team)))
        blue_team = self.owner.rng.choice(
            diff_list, self.owner.nums_comp_pairs, replace=False
        )

        return rung, population_range, red_team.tolist(), blue_team.tolist()

    def _mutate_population(
        self,
        red_team: Sequence[int],
        blue_team: Sequence[int],
        rung: dict,
        population_range: int,
        fidelity: int | float,
    ) -> tuple[list[Trial], np.ndarray]:
        """Get the mutated population and hurdles"""
        winner_list = []
        loser_list = []

        if set(red_team) != set(blue_team):
            hurdles = np.zeros(1)
            for i, _ in enumerate(red_team):
                winner, loser = (
                    (red_team, blue_team)
                    if self.owner.performance[red_team[i]]
                    < self.owner.performance[blue_team[i]]
                    else (blue_team, red_team)
                )

                winner_list.append(winner[i])
                loser_list.append(loser[i])
                hurdles += self.owner.performance[winner[i]]
                self._mutate(winner[i], loser[i])

            hurdles /= len(red_team)
            self.owner.hurdles.append(hurdles)

            logger.debug("Evolution hurdles are: %s", str(self.owner.hurdles))

        trials = []
        trial_ids = set()
        nums_all_equal = [0] * population_range
        for i in range(population_range):
            point: Sequence[int | float] = [0] * len(self.space)
            while True:
                point = list(point)
                point[
                    list(self.space.keys()).index(self.owner.fidelity_index)
                ] = fidelity

                for j in self.search_space_without_fidelity:
                    point[j] = self.owner.population[j][i]

                trial = format_trials.tuple_to_trial(point, self.space)
                trial_id = self.owner.get_id(trial)

                if trial_id in trial_ids:
                    nums_all_equal[i] += 1
                    logger.debug("find equal one, continue to mutate.")
                    self._mutate(i, i)
                elif self.owner.has_suggested(trial):
                    nums_all_equal[i] += 1
                    logger.debug("find one already suggested, continue to mutate.")
                    self._mutate(i, i)
                else:
                    break
                if nums_all_equal[i] > self.owner.max_retries:
                    logger.warning(
                        "Can not Evolve any more. You can make an early stop."
                    )
                    break

            if nums_all_equal[i] < self.owner.max_retries:
                trials.append(trial)
                trial_ids.add(trial_id)
            else:
                logger.debug("Dropping trial %s", trial)

        return trials, np.array(nums_all_equal)

    def get_candidates(self, rung_id: int) -> list[Trial]:
        """Get a candidate for promotion"""
        if rung_id not in self._candidates:
            rung, population_range, red_team, blue_team = self._get_teams(rung_id)
            fidelity = self.rungs[rung_id + 1]["resources"]
            self._candidates[rung_id] = self._mutate_population(
                red_team, blue_team, rung, population_range, fidelity
            )[0]

        candidates: list[Trial] = []
        for candidate in self._candidates[rung_id]:
            if not self.owner.has_suggested(candidate):
                candidates.append(candidate)
        return candidates

    def _mutate(self, winner_id: int, loser_id: int) -> None:
        select_genes_key_list = self.owner.rng.choice(
            self.search_space_without_fidelity,
            self.owner.nums_mutate_gene,
            replace=False,
        )
        self.copy_winner(winner_id, loser_id)
        kwargs = copy.deepcopy(self.mutate_attr)

        for i, _ in enumerate(select_genes_key_list):
            space = self.space.values()[select_genes_key_list[i]]
            old = self.owner.population[select_genes_key_list[i]][loser_id]
            new = self.mutate_func(space, self.owner.rng, old, **kwargs)
            self.owner.population[select_genes_key_list[i]][loser_id] = new

        self.owner.performance[loser_id] = -1

    def copy_winner(self, winner_id: int, loser_id: int) -> None:
        """Copy winner to loser"""
        for key in self.search_space_without_fidelity:
            self.owner.population[key][loser_id] = self.owner.population[key][winner_id]
