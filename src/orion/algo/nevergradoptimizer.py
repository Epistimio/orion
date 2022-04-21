"""
:mod:`orion.algo.nevergrad.nevergradoptimizer -- TODO
=================================================

TODO: Write long description
"""
import pickle

try:
    import nevergrad as ng

    IMPORT_ERROR = None

except ImportError as err:
    IMPORT_ERROR = err

from orion.algo.base import BaseAlgorithm
from orion.core.utils.format_trials import dict_to_trial


class SpaceConverter(dict):
    """Convert Orion's search space to a different format."""

    def register(self, typ, prior):
        """Register a conversion function for the given type and prior."""

        def deco(func):
            self[typ, prior] = func
            return func

        return deco

    def __call__(self, space):
        try:
            return ng.p.Instrumentation(
                **{
                    name: self[dim.type, dim.prior_name](self, dim)
                    for name, dim in space.items()
                }
            )
        except KeyError as exc:
            raise KeyError(
                f"Dimension with type and prior: {exc.args[0]} cannot be converted to nevergrad."
            )


to_ng_space = SpaceConverter()


def _intshape(shape):
    # ng.p.Array does not accept np.int64 in shapes, they have to be ints
    return tuple(int(x) for x in shape)


@to_ng_space.register("categorical", "choices")
def _(_, dim):
    if dim.shape:
        raise NotImplementedError("Array of Categorical cannot be converted.")
    if len(set(dim.original_dimension.prior.pk)) != 1:
        raise NotImplementedError(
            "All categories in Categorical must have the same probability."
        )
    return ng.p.Choice(dim.interval())


@to_ng_space.register("real", "uniform")
def _(_, dim):
    lower, upper = dim.interval()
    if dim.shape:
        return ng.p.Array(lower=lower, upper=upper, shape=_intshape(dim.shape))
    else:
        return ng.p.Scalar(lower=lower, upper=upper)


@to_ng_space.register("integer", "int_uniform")
def _(self, dim):
    return self["real", "uniform"](self, dim).set_integer_casting()


@to_ng_space.register("real", "reciprocal")
def _(_, dim):
    if dim.shape:
        raise NotImplementedError("Array with reciprocal prior cannot be converted.")
    lower, upper = dim.interval()
    return ng.p.Log(lower=lower, upper=upper, exponent=2)


@to_ng_space.register("integer", "int_reciprocal")
def _(self, dim):
    return self["real", "reciprocal"](self, dim).set_integer_casting()


@to_ng_space.register("real", "norm")
def _(_, dim):
    if dim.shape:
        raise NotImplementedError("Array with normal prior cannot be converted.")
    return ng.p.Scalar(init=dim.original_dimension.prior.mean()).set_mutation(
        sigma=dim.original_dimension.prior.std()
    )


@to_ng_space.register("fidelity", "None")
def _(_, dim):
    if dim.shape:
        raise NotImplementedError("Array of Fidelity cannot be converted.")
    _, upper = dim.interval()
    # No equivalent to Fidelity space, so we always use the upper value
    return upper


NOT_WORKING = {
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
}


class NevergradOptimizer(BaseAlgorithm):
    """TODO: Class docstring

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``

    """

    requires_type = None
    requires_dist = None
    requires_shape = None

    def __init__(
        self, space, model_name="NGOpt", seed=None, budget=100, num_workers=10
    ):
        if IMPORT_ERROR:
            raise IMPORT_ERROR

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
        super().__init__(
            space,
            model_name=model_name,
            seed=seed,
            budget=budget,
            num_workers=num_workers,
        )

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
        state_dict["algo"] = pickle.dumps(self.algo)
        state_dict["_is_done"] = self._is_done
        state_dict["_fresh"] = self._fresh
        state_dict["_trial_mapping"] = {
            tid: list(sugg) for tid, sugg in self._trial_mapping.items()
        }
        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)
        self.algo = pickle.loads(state_dict["algo"])
        self._is_done = state_dict["_is_done"]
        self._fresh = state_dict["_fresh"]
        self._trial_mapping = state_dict["_trial_mapping"]

    def _associate_trial(self, trial, suggestion):
        """Associate a trial with a Nevergrad suggestion.

        Returns True if the trial was not seen before, false otherwise.
        """
        tid = self.get_id(trial)
        seen = tid in self._trial_mapping
        if seen:
            orig_trial, existing = self._trial_mapping[tid]
            if orig_trial.status == "completed":
                self.algo.tell(suggestion, orig_trial.objective.value)
            else:
                existing.append(suggestion)

        else:
            self._trial_mapping[tid] = (trial, [suggestion])
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
            return None

    def _can_produce(self):
        if self.is_done:
            return False

        algo = self.algo
        is_sequential = algo.no_parallelization
        if not is_sequential and hasattr(algo, "optim"):
            is_sequential = algo.optim.no_parallelization

        if is_sequential and algo.num_ask > (algo.num_tell - algo.num_tell_not_asked):
            return False

        return True

    def suggest(self, num):
        """Suggest a `num`ber of new sets of parameters.

        TODO: document how suggest work for this algo

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
        attempts = 0
        max_attempts = num + 100
        trials = []
        while len(trials) < num and attempts < max_attempts and self._can_produce():
            attempts += 1
            trial = self._ask()
            if trial is not None:
                trials.append(trial)

        if not trials and self._can_produce():
            self._is_done = True

        self._fresh = False
        return trials

    def observe(self, trials):
        """Observe the `trials` new state of result.

        TODO: document how observe work for this algo

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
    def is_done(self):
        return self._is_done or super().is_done
