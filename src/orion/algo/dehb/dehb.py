"""
:mod:`orion.algo.dehb.dehb -- DEHB
==================================

Module for the wrapper around DEHB: https://github.com/automl/DEHB.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from typing import ClassVar, NamedTuple

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.dehb.brackets import SHBracketManager
from orion.algo.space import Fidelity, Space
from orion.core.utils import format_trials
from orion.core.utils.module_import import ImportOptional
from orion.core.worker.trial import Trial

with ImportOptional("DEHB") as import_optional:
    from dehb.optimizers import DEHB as DEHBImpl
    from sspace.convert import convert_space
    from sspace.convert import transform as to_orion

if import_optional.failed:

    # pylint: disable=function-redefined,too-few-public-methods
    class DEHBImpl:  # noqa: F811
        """Dummy implementation for optional imports"""


logger = logging.getLogger(__name__)


class UnsupportedConfiguration(Exception):
    """Raised when an unsupported configuration is sent"""


SPACE_ERROR = """
DEHB cannot be used if space does not contain a fidelity dimension.
"""

MUTATION_STRATEGIES = [
    "rand1",
    "rand2dir",
    "randtobest1",
    "currenttobest1",
    "best1",
    "best2",
    "rand2",
]

CROSSOVER_STRATEGY = [
    "bin",
    "exp",
]

FIX_MODES = ["random", "clip"]


class RungResult(NamedTuple):
    """NamedTuple"""

    cost: int | float
    fitness: float


class _CustomDEHBImpl(DEHBImpl):
    def __init__(self, duplicates, **kwargs):
        self.duplicates = duplicates
        super().__init__(**kwargs)

    def f_objective(self, *args, **kwargs) -> None:
        """Not needed for Orion, the objective is called by the worker"""

    def _start_new_bracket(self) -> SHBracketManager:
        """Starts a new bracket based on Hyperband"""
        # start new bracket
        self.iteration_counter += (
            1  # iteration counter gives the bracket count or bracket ID
        )
        n_configs, budgets = self.get_next_iteration(self.iteration_counter)
        bracket = SHBracketManager(
            n_configs=n_configs,
            budgets=budgets,
            bracket_id=self.iteration_counter,
            duplicates=self.duplicates,
        )
        self.active_brackets.append(bracket)
        return bracket

    def register_job(self, job_info: dict) -> None:
        """Register to DEHB's backend"""

        # pass information of job submission to Bracket Manager
        for bracket in self.active_brackets:
            if bracket.bracket_id == job_info["bracket_id"]:
                # registering is IMPORTANT for Bracket Manager to perform SH
                bracket.register_job(job_info["budget"])
                break

    def init_population(self, pop_size: int) -> list[numpy.ndarray]:
        """Generate our initial population of sample

        Parameters
        ----------
        pop_size: int
            Number of samples to generate

        """
        population = self.cs.sample_configuration(size=pop_size)
        population = [
            self.configspace_to_vector(individual) for individual in population
        ]
        return population

    def observe(self, job_info: dict, cost: int | float, fitness: float) -> None:
        """Observe a completed job"""
        config: numpy.ndarray = job_info["config"]
        budget: int | float = job_info["budget"]
        parent_id: int = job_info["parent_id"]
        bracket_id: int = job_info["bracket_id"]

        for bracket in self.active_brackets:
            if bracket.bracket_id == bracket_id:
                # bracket job complete
                # IMPORTANT to perform synchronous SH
                bracket.complete_job(budget)

        # carry out DE selection
        if fitness <= self.de[budget].fitness[parent_id]:
            self.de[budget].population[parent_id] = config
            self.de[budget].fitness[parent_id] = fitness

        # NOTE: This dictionary seams useless in dehb source code.
        info: dict = {}
        # updating incumbents
        if self.de[budget].fitness[parent_id] < self.inc_score:
            self._update_incumbents(
                config=self.de[budget].population[parent_id],
                score=self.de[budget].fitness[parent_id],
                info=info,
            )

        # book-keeping
        self._update_trackers(
            traj=self.inc_score,
            runtime=cost,
            history=(config.tolist(), float(fitness), float(cost), float(budget), info),
        )

    def seed_rng(self, seed: int) -> None:
        """Seed the state of rngs used by DEHB"""
        numpy.random.seed(seed)
        self.cs.seed(numpy.random.randint(numpy.iinfo(numpy.int32).max))
        # Reset sub population
        self.reset()

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        state = dict(self.__dict__)
        state["client"] = None
        state["logger"] = None
        for key in [
            "active_brackets",
            "iteration_counter",
            "de",
            "_max_pop_size",
            "start",
            "traj",
            "runtime",
            "history",
        ]:
            state[key] = getattr(self, key, None)

        return deepcopy(
            {
                "state": state,
                "numpy_GlobalState": numpy.random.get_state(),
                "numpy_RandomState": self.cs.random.get_state(),
            }
        )

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of DEHB"""
        for k, v in state_dict["state"].items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.error("DEHB does not have attribute %s", k)

        numpy.random.set_state(state_dict["numpy_GlobalState"])
        self.cs.random.set_state(state_dict["numpy_RandomState"])


# pylint: disable=too-many-public-methods
class DEHB(BaseAlgorithm):
    """Differential Evolution with HyperBand

    This class is a wrapper around the library DEHB:
    https://github.com/automl/DEHB.

    For more information on the algorithm,
    see original paper at https://arxiv.org/abs/2105.09821.

    Awad, Noor, Neeratyoy Mallik, and Frank Hutter. "Dehb: Evolutionary hyperband for scalable,
    robust and efficient hyperparameter optimization." arXiv preprint arXiv:2105.09821 (2021).

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.

    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``

    mutation_factor: float
        Mutation probability
        Default: ``0.5``

    crossover_prob: float
        Crossover probability
        Default: ``0.5``

    mutation_strategy: str
        Mutation strategy rand1, rand2dir randtobest1 currenttobest1 best1 best2 rand2
        Default: ``'rand1'``

    crossover_strategy: str
        Crossover strategy bin or exp
        Default: ``'bin'``

    boundary_fix_type: str
        Boundary fix method, clip or random
        Default: ``'random'``

    min_clip: float
        Min clip when boundary fix method is clip
        Default: ``None``

    max_clip: float
        Max clip when boundary fix method is clip
        Default: ``None``

    """

    requires_type: ClassVar[str | None] = None
    requires_dist: ClassVar[str | None] = None
    requires_shape: ClassVar[str | None] = "flattened"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        space: Space,
        seed: int | None = None,
        mutation_factor: float = 0.5,
        crossover_prob: float = 0.5,
        mutation_strategy: str = "rand1",
        crossover_strategy: str = "bin",
        boundary_fix_type: str = "random",
        min_clip: int | None = None,
        max_clip: int | None = None,
    ):
        import_optional.ensure()

        # Sanity Check
        if mutation_strategy not in MUTATION_STRATEGIES:
            raise UnsupportedConfiguration(
                f"Mutation strategy {mutation_strategy} not supported"
            )

        if crossover_strategy not in CROSSOVER_STRATEGY:
            raise UnsupportedConfiguration(
                f"Crossover strategy {crossover_strategy} not supported"
            )

        if boundary_fix_type not in FIX_MODES:
            raise UnsupportedConfiguration(
                f"Boundary fix type {boundary_fix_type} not supported"
            )

        super().__init__(
            space,
            seed=seed,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            mutation_strategy=mutation_strategy,
            crossover_strategy=crossover_strategy,
            boundary_fix_type=boundary_fix_type,
            min_clip=min_clip,
            max_clip=max_clip,
        )

        # Extract fidelity information
        if self.fidelity_index is None:
            raise RuntimeError(SPACE_ERROR)

        fidelity_dim: Fidelity = space[self.fidelity_index]

        # NOTE: This isn't a Fidelity, it's a TransformedDimension<Fidelity>
        from orion.core.worker.transformer import TransformedDimension

        # NOTE: Currently bypassing (possibly more than one) `TransformedDimension` wrappers to get
        # the 'low', 'high' and 'base' attributes.
        while isinstance(fidelity_dim, TransformedDimension):
            fidelity_dim = fidelity_dim.original_dimension
        assert isinstance(fidelity_dim, Fidelity)

        self.rung: int | None = None
        self.seed = seed
        self.job_infos: defaultdict[str, list] = defaultdict(list)
        self.job_results: dict[str, RungResult] = {}
        self.duplicates: defaultdict[str, int] = defaultdict(int)

        configspace = convert_space(self.space)

        # Initialize
        self.dehb = _CustomDEHBImpl(
            duplicates=self.duplicates,
            cs=configspace,
            configspace=True,
            dimensions=len(configspace.get_hyperparameters()),
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            strategy=f"{mutation_strategy}_{crossover_strategy}",
            min_clip=min_clip,
            max_clip=max_clip,
            boundary_fix_type=boundary_fix_type,
            max_age=numpy.inf,
            # Derived
            min_budget=fidelity_dim.low,
            max_budget=fidelity_dim.high,
            eta=fidelity_dim.base,
            # Disable their Dask Integration
            n_workers=1,
            client=None,
            # No need for the user function
            f=None,
        )
        self.seed_rng(self.seed)
        self.rung = len(self.dehb.budgets)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super().state_dict
        state_dict["DEHB_statedict"] = self.dehb.state_dict

        return deepcopy(state_dict)

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)
        self.dehb.set_state(state_dict["DEHB_statedict"])

    def seed_rng(self, seed: int | None) -> None:
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int or None
            Integer seed for the random number generator.

        """
        if hasattr(self, "dehb"):
            self.dehb.seed_rng(seed)

    @property
    def is_done(self) -> bool:
        """Return True, if an algorithm holds that there can be no further improvement."""
        # pylint: disable=protected-access
        return self.dehb._is_run_budget_exhausted(None, self.rung, None)

    def sample_to_trial(self, sample: numpy.ndarray, fidelity: int) -> Trial:
        """Convert a ConfigSpace sample into a trial"""
        config = self.dehb.vector_to_configspace(sample)
        hps = {}

        for k, v in self.space.items():

            if v.type == "fidelity":
                hps[k] = fidelity

            else:
                hps[k] = config[k]

        return format_trials.dict_to_trial(to_orion(hps), self.space)

    def suggest(self, num: int) -> list[Trial]:
        """Suggest a number of new sets of parameters.

        Parameters
        ----------
        num: int, optional
            Number of trials to suggest. The algorithm may return less than the number of trials
            requested.

        Returns
        -------
        list of trials or None
            A list of trials representing values suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """
        trials: Trial = []
        while len(trials) < num:
            if self.is_done:
                break

            # pylint: disable=protected-access
            job_info: dict = self.dehb._get_next_job()
            job_info["done"] = 0

            # We are generating trials for a bracket that is too high
            if self.rung is not None and job_info["bracket_id"] >= self.rung:
                break

            # Generate Orion trial
            new_trial = self.sample_to_trial(
                job_info["config"], fidelity=job_info["budget"]
            )

            # DEHB may sample 2 identical trials if working in a purely discrete spaceI
            # It this case you will have has_suggested(new_trial) is True, and you will
            # discard this trial.
            # It's fine to discard the trial, but we should keep track of the job_info.
            # DEHB does not know that we discarded the trial and will be waiting for the
            # result. We need to keep track of both job_info so that when we have
            # the result of the first trial, we assign it to the second job_info as well.

            if not self.has_suggested(new_trial):
                # Store metadata
                self.job_infos[self.get_id(new_trial)].append(job_info)
                self.dehb.register_job(job_info)

                # Standard Orion
                self.register(new_trial)
                trials.append(new_trial)
                logger.debug("Suggest new trials %s", new_trial)
            else:
                logger.debug("Already suggested %s", new_trial)

                # Do we already have a result for this trial ?
                result = self.job_results.get(self.get_id(new_trial))

                # Keep track of duplicated jobs per brackets
                # Bracket is only done after we reach the budget for unique
                # jobs
                self.duplicates[str(job_info["budget"])] += 1

                # if so observe it right now and discard
                if result is not None:
                    self.dehb.observe(job_info, *result)
                else:
                    # else we need to keep track of it to observe it later
                    self.job_infos[self.get_id(new_trial)].append(job_info)

        return trials

    def observe(self, trials: list[Trial]) -> None:
        """Observe the `trials` new state of result.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        for trial in trials:
            if not self.has_suggested(trial):
                logger.debug("Ignore unseen trial %s", trial)
                continue

            if self.has_observed(trial):
                logger.debug("Ignore already observed trial %s", trial)
                continue

            self.register(trial)

            if trial.status == "completed":
                self.observe_one(trial)
                logger.debug(
                    "Observe trial %s (Remaining %d)", trial, len(self.job_infos)
                )

    def observe_one(self, trial: Trial) -> None:
        """Observe a single trial"""

        # Get all the job sampled by DEHB, it might be more than one
        job_infos = self.job_infos.get(self.get_id(trial), [])

        if not job_infos:
            # this should be 100% unreachable because we check
            # if the trial was suggested inside `observe`
            logger.error("Could not find trial %s", self.get_id(trial))
            return

        # Yes, it is odd; fidelity is cost and fitness is objective
        cost = trial.params[self.fidelity_index]
        fitness = trial.objective.value

        # Store the result for later, if we sample
        # a trial that is too alike for us to evaluate
        self.job_results[self.get_id(trial)] = RungResult(cost, fitness)

        for job_info in job_infos:
            cost = job_info["budget"]
            self.dehb.observe(job_info, cost, fitness)
