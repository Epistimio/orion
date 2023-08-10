"""
Nevergrad Optimizer
===================

Wraps the nevergrad library to expose its algorithm to orion

"""
from __future__ import annotations

import logging
import pickle
from typing import Callable, Iterable, Sequence, SupportsInt

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Categorical, Dimension, Fidelity, Integer, Real, Space
from orion.core.utils.format_trials import dict_to_trial
from orion.core.utils.module_import import ImportOptional
from orion.core.worker.trial import Trial

with ImportOptional("nevergrad") as import_optional:
    import nevergrad as ng
    from nevergrad.parametrization.container import Instrumentation
    from nevergrad.parametrization.core import Parameter


logger = logging.getLogger(__name__)

registry: dict[tuple[str, str], Callable[[Dimension], Parameter]] = {}


def register(dimension_type: str, prior: str):
    """Register a conversion function for the given type and prior."""

    def deco(func):
        registry[dimension_type, prior] = func
        return func

    return deco


def to_ng_space(orion_space: Space) -> Instrumentation:
    """Convert an orion space to a nevergrad space."""
    import_optional.ensure()
    converted_dimensions: dict[str, Parameter] = {}
    for name, dim in orion_space.items():
        try:
            converted_dimensions[name] = registry[dim.type, dim.prior_name](dim)
        except KeyError as exc:
            raise RuntimeError(
                f"Dimension with type and prior: {exc.args[0]} cannot be converted to nevergrad."
            ) from exc

    return ng.p.Instrumentation(**converted_dimensions)


def _intshape(shape: Iterable[SupportsInt]) -> tuple[int, ...]:
    # ng.p.Array does not accept np.int64 in shapes, they have to be ints
    return tuple(int(x) for x in shape)


@register("categorical", "choices")
def _(dim: Categorical):
    if dim.shape:
        raise NotImplementedError("Array of Categorical cannot be converted.")
    if len(set(dim.original_dimension.prior.pk)) != 1:
        raise NotImplementedError(
            "All categories in Categorical must have the same probability."
        )
    return ng.p.Choice(dim.interval())


@register("real", "uniform")
def _(dim: Dimension):
    lower, upper = dim.interval()
    if dim.shape:
        # Temporary fix pending [#800]
        # ng.p.Array expects an array with a shape or a float
        # an array with an empty shape is not valid
        # so we cast it to a float
        if hasattr(lower, "shape") and lower.shape == ():
            lower = float(lower)

        if hasattr(upper, "shape") and upper.shape == ():
            upper = float(upper)

        return ng.p.Array(lower=lower, upper=upper, shape=_intshape(dim.shape))
    else:
        return ng.p.Scalar(lower=lower, upper=upper)


@register("integer", "int_uniform")
def _(dim: Integer):
    return registry["real", "uniform"](dim).set_integer_casting()


@register("integer", "int_reciprocal")
def _(dim: Integer):
    return registry["real", "reciprocal"](dim).set_integer_casting()


@register("real", "reciprocal")
def _(dim: Real):
    if dim.shape:
        raise NotImplementedError("Array with reciprocal prior cannot be converted.")
    lower, upper = dim.interval()
    return ng.p.Log(lower=lower, upper=upper, exponent=2)


@register("real", "norm")
@register("real", "normal")
def _(dim: Real):
    if dim.shape:
        raise NotImplementedError("Array with normal prior cannot be converted.")
    return ng.p.Scalar(init=dim.original_dimension.prior.mean()).set_mutation(
        sigma=dim.original_dimension.prior.std()
    )


@register("fidelity", "None")
def _(dim: Fidelity):
    if dim.shape:
        raise NotImplementedError("Array of Fidelity cannot be converted.")
    _, upper = dim.interval()
    # No equivalent to Fidelity space, so we always use the upper value
    return upper


NOT_WORKING = {
    "ASCMADEthird",
    "BO",
    "BOSplit",
    "BayesOptimBO",
    "DiscreteDoerrOnePlusOne",
    "NelderMead",
    "PCABO",
    "SPSA",
    "SQP",
    "NoisyBandit",
    "NoisyOnePlusOne",
    "OptimisticDiscreteOnePlusOne",
    "OptimisticNoisyOnePlusOne",
    "Powell",
    "CmaFmin2",
    "RPowell",
    "RSQP",
    "PymooNSGA2",
    "NGOpt12",
    "NGOpt13",
    "NGOpt14",
    "NGOpt21",
    "NGOpt38",
}


class NevergradOptimizer(BaseAlgorithm):
    """Wraps the nevergrad library to expose its algorithm to orion

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    model_name: str
        Nevergrad model to use as optimizer
    budget: int
        Maximal number of trial to generated
    num_workers: int
        Number of worker to use
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``

    """

    requires_type = None
    requires_dist = None
    requires_shape = None

    def __init__(
        self,
        space: Space,
        model_name: str = "NGOpt",
        seed: int | Sequence[int] | None = None,
        budget: int = 100,
        num_workers: int = 10,
    ):
        import_optional.ensure()

        super().__init__(space)
        self.model_name = model_name
        self.seed = seed
        self.budget = budget
        self.num_workers = num_workers

        if model_name in NOT_WORKING:
            raise ValueError(f"Model {model_name} is not supported.")

        param = to_ng_space(space)
        self.algo = ng.optimizers.registry[model_name](
            parametrization=param, budget=budget, num_workers=num_workers
        )
        self.algo.enable_pickling()
        self._trial_mapping = {}
        self._fresh = True
        self._is_done = False

        self.seed = seed
        if seed is not None:
            self.seed_rng(seed)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int
            Integer seed for the random number generator.

        """
        self.algo.parametrization.random_state.seed(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super().state_dict
        state_dict["algo"] = pickle.dumps(self.algo)  # type: ignore
        state_dict["_is_done"] = self._is_done
        state_dict["_fresh"] = self._fresh
        state_dict["_trial_mapping"] = {
            trial_id: list(suggestions)
            for trial_id, suggestions in self._trial_mapping.items()
        }
        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        Parameters
        ----------
        state_dict: dict
            Dictionary representing state of an algorithm

        """
        super().set_state(state_dict)
        self.algo = pickle.loads(state_dict["algo"])
        self._is_done = state_dict["_is_done"]
        self._fresh = state_dict["_fresh"]
        self._trial_mapping = state_dict["_trial_mapping"]

    def _associate_trial(self, trial: Trial, suggestion: Parameter):
        """Associate a trial with a Nevergrad suggestion.

        Returns
        -------
        True if the trial was not seen before, false otherwise.

        """
        trial_id = self.get_id(trial)
        seen = trial_id in self._trial_mapping
        if seen:
            orig_trial, existing = self._trial_mapping[trial_id]
            if orig_trial.status == "completed":
                self.algo.tell(suggestion, orig_trial.objective.value)
            else:
                existing.append(suggestion)

        else:
            self._trial_mapping[trial_id] = (trial, [suggestion])

        return not seen

    def _ask(self):
        suggestion = self.algo.ask()
        if suggestion.args:
            raise RuntimeError(
                "Nevergrad sampled a trial with args but this should never happen."
                " Please report this issue at"
                " https://github.com/Epistimio/orion.algo.nevergrad/issues"
            )
        new_trial = dict_to_trial(suggestion.kwargs, self.space)

        if self._associate_trial(new_trial, suggestion):
            self.register(new_trial)
            return new_trial
        else:
            logger.debug("Ignoring duplicated trial")
            return None

    def _can_produce(self):
        if self.is_done:
            return False

        algo = self.algo
        is_sequential = algo.no_parallelization

        if not is_sequential and hasattr(algo, "optim"):
            is_sequential = algo.optim.no_parallelization

        if is_sequential and algo.num_ask > (algo.num_tell - algo.num_tell_not_asked):
            logger.debug(
                "Cannot produce new trials because %d > (%d - %d)",
                algo.num_ask,
                algo.num_tell,
                algo.num_tell_not_asked,
            )
            return False

        return True

    def suggest(self, num: int) -> list[Trial]:
        """Suggest a number of new sets of parameters.

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
            trials to complete), in which case it will return None.

        """
        attempts = 0
        max_attempts = num + 100
        trials: list[Trial] = []

        while len(trials) < num and attempts < max_attempts and self._can_produce():
            attempts += 1

            trial = self._ask()
            if trial is not None:
                trials.append(trial)

        if not trials and self._can_produce():
            self._is_done = True

        self._fresh = False
        return trials

    def observe(self, trials: list[Trial]) -> None:
        """Observe the trials new state of result.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        for trial in trials:
            if trial.status == "completed":
                tid = self.get_id(trial)
                if tid in self._trial_mapping:
                    _, suggestions = self._trial_mapping[tid]
                else:
                    sugg = self.algo.parametrization.spawn_child(((), trial.params))
                    suggestions = [sugg]
                for suggestion in suggestions:
                    self.algo.tell(suggestion, trial.objective.value)
                self._trial_mapping[tid] = (trial, [])
                self._fresh = True

        super().observe(trials)

    @property
    def is_done(self) -> bool:
        return self._is_done or super().is_done
