"""
:mod:`orion.algo.bohb` -- BOHB
==============================

Module for the wrapper around HpBandSter.
"""
import copy

import numpy as np

from orion.algo.base import BaseAlgorithm
from orion.algo.base.parallel_strategy import strategy_factory
from orion.algo.space import Fidelity
from orion.core.utils.format_trials import dict_to_trial
from orion.core.utils.module_import import ImportOptional

with ImportOptional("BOHB") as import_optional:
    from hpbandster.optimizers.config_generators.bohb import BOHB as CG_BOHB
    from hpbandster.optimizers.iterations import SuccessiveHalving
    from sspace.convert import convert_space, reverse, transform

if import_optional.failed:
    CG_BOHB = None  # noqa: F811
    # pylint: disable=invalid-name
    SuccessiveHalving = None  # noqa: F811


SPACE_ERROR = """
BOHB cannot be used if space does not contain a fidelity dimension.
For more information on the configuration and usage of BOHB, see
https://orion.readthedocs.io/en/develop/user/algorithms.html#bohb-algorithm
"""


# SuccessiveHalving gives us tuples of stuff to run but expects the results
# to be packaged up in jobs so this is filling in for those jobs.
class FakeJob:  # pylint: disable=too-few-public-methods
    """
    Minimal HpBandSter Job mock.

    This mimics enough of the HpBandSter Job interface to report results.
    """

    def __init__(self, run, trial):
        self.id = run[0]  # pylint: disable=invalid-name
        self.kwargs = dict(config=reverse(run[1]), budget=run[2])
        self.timestamps = {}
        self.result = {}
        if trial.objective is not None:
            self.result["loss"] = trial.objective.value
        self.exception = None


class BOHB(BaseAlgorithm):
    """Bayesian Optimization with HyperBand

    This class is a wrapper around the library HpBandSter:
    https://github.com/automl/HpBandSter.

    For more information on the algorithm,
    see original paper at https://arxiv.org/abs/1807.01774.

    Falkner, Stefan, Aaron Klein, and Frank Hutter. "BOHB: Robust and efficient hyperparameter
    optimization at scale." In International Conference on Machine Learning, pp. 1437-1446. PMLR,
    2018.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    min_points_in_model: int
        Number of observations to start building a KDE. If ``None``, uses number
        of dimensions in the search space + 1. Default: ``None``
    top_n_percent: int
        Percentage ( between 1 and 99) of the observations that are considered good. Default: 15
    num_samples: int
        Number of samples to optimize Expected Improvement. Default: 64
    random_fraction: float
        Fraction of purely random configurations that are sampled from the
        prior without the model. Default: 1/3
    bandwidth_factor: float
        To encourage diversity, the points proposed to optimize EI, are sampled
        from a 'widened' KDE where the bandwidth is multiplied by this factor. Default: 3
    min_bandwidth: float
        To keep diversity, even when all (good) samples have the same value
        for one of the parameters, a minimum bandwidth is used instead of
        zero. Default: 1e-3
    parallel_strategy: dict or None, optional
        The configuration of a parallel strategy to use for pending trials or broken trials.
        Default is a MaxParallelStrategy for broken trials and NoParallelStrategy for pending
        trials.

    """

    requires_type = None
    requires_dist = None
    requires_shape = "flattened"

    def __init__(
        self,
        space,
        seed=None,
        min_points_in_model=None,
        top_n_percent=15,
        num_samples=64,
        random_fraction=1 / 3,
        bandwidth_factor=3,
        min_bandwidth=1e-3,
        parallel_strategy=None,
    ):  # pylint: disable=too-many-arguments
        import_optional.ensure()

        if parallel_strategy is None:
            parallel_strategy = {
                "of_type": "StatusBasedParallelStrategy",
                "strategy_configs": {
                    "broken": {
                        "of_type": "MaxParallelStrategy",
                    },
                },
            }

        self.strategy = strategy_factory.create(**parallel_strategy)

        super().__init__(
            space,
            seed=seed,
            min_points_in_model=min_points_in_model,
            top_n_percent=top_n_percent,
            num_samples=num_samples,
            random_fraction=random_fraction,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth,
            parallel_strategy=parallel_strategy,
        )
        self.trial_meta = {}
        self.trial_results = {}
        self.iteration = 0
        self.iterations = []

        fidelity_index = self.fidelity_index
        if fidelity_index is None:
            raise RuntimeError(SPACE_ERROR)
        fidelity_dim = self.space[fidelity_index]

        fidelity_dim: Fidelity = space[self.fidelity_index]

        # NOTE: This isn't a Fidelity, it's a TransformedDimension<Fidelity>
        from orion.core.worker.transformer import TransformedDimension

        # NOTE: Currently bypassing (possibly more than one) `TransformedDimension` wrappers to get
        # the 'low', 'high' and 'base' attributes.
        while isinstance(fidelity_dim, TransformedDimension):
            fidelity_dim = fidelity_dim.original_dimension
        assert isinstance(fidelity_dim, Fidelity)

        self.min_budget = fidelity_dim.low
        self.max_budget = fidelity_dim.high
        self.eta = fidelity_dim.base

        self._setup()

    def _setup(self):
        self.max_sh_iter = (
            -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta)) + 1
        )
        self.budgets = self.max_budget * np.power(
            self.eta, -np.linspace(self.max_sh_iter - 1, 0, self.max_sh_iter)
        )

        self.bohb = CG_BOHB(  # pylint: disable=attribute-defined-outside-init
            configspace=convert_space(self.space),
            min_points_in_model=self.min_points_in_model,
            top_n_percent=self.top_n_percent,
            num_samples=self.num_samples,
            random_fraction=self.random_fraction,
            bandwidth_factor=self.bandwidth_factor,
            min_bandwidth=self.min_bandwidth,
        )
        self.bohb.configspace.seed(self.seed)

    def _make_iteration(self, iteration):
        ss = self.max_sh_iter - 1 - (iteration % self.max_sh_iter)

        # number of configurations in that bracket
        n0 = int(np.floor((self.max_sh_iter) / (ss + 1)) * self.eta**ss)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(ss + 1)]

        return SuccessiveHalving(
            HPB_iter=iteration,
            num_configs=ns,
            budgets=self.budgets[(-ss - 1) :],
            config_sampler=self.bohb.get_config,
        )

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int
            Integer seed for the random number generator.

        """
        np.random.seed(seed)
        if hasattr(self, "bohb"):
            self.bohb.configspace.seed(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super().state_dict
        state_dict["rng_state"] = np.random.get_state()
        state_dict["eta"] = self.eta
        state_dict["min_budget"] = self.min_budget
        state_dict["max_budget"] = self.max_budget
        state_dict["iteration"] = self.iteration
        state_dict["iterations"] = copy.deepcopy(self.iterations)
        state_dict["trial_meta"] = dict(self.trial_meta)
        state_dict["trial_results"] = dict(self.trial_results)
        state_dict["bohb"] = copy.deepcopy(self.bohb)
        state_dict["strategy"] = self.strategy.state_dict
        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)
        np.random.set_state(state_dict["rng_state"])
        self.eta = state_dict["eta"]
        self.min_budget = state_dict["min_budget"]
        self.max_budget = state_dict["max_budget"]
        self.iteration = state_dict["iteration"]
        self.iterations = state_dict["iterations"]
        self.trial_meta = state_dict["trial_meta"]
        self.trial_results = state_dict["trial_results"]
        self.bohb = state_dict["bohb"]  # pylint: disable=attribute-defined-outside-init
        self.strategy.set_state(state_dict["strategy"])
        self._setup()

    def suggest(self, num):
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

        def run_to_trial(run):
            params = transform(run[1])
            params[self.fidelity_index] = run[2]
            return dict_to_trial(params, self.space)

        def sample_iteration(iteration, trials):
            while len(trials) < num and not iteration.is_finished:
                run = iteration.get_next_run()
                if run is None:
                    break
                new_trial = run_to_trial(run)

                # This means the job was already suggested and we have a result
                result = self.trial_results.get(self.get_id(new_trial), None)
                if result is not None:
                    job = FakeJob(run, new_trial)
                    job.result["loss"] = result
                    iteration.register_result(job)
                    self.bohb.new_result(job)
                    continue

                self.trial_meta.setdefault(self.get_id(new_trial), []).append(run)
                self.register(new_trial)
                trials.append(new_trial)

        trials = []
        for it in self.iterations:
            sample_iteration(it, trials)
        # If we don't have enough trials and there are still
        # some iterations left
        if self.iteration < len(self.budgets):
            self.iterations.append(self._make_iteration(self.iteration))
            self.iteration += 1
            sample_iteration(self.iterations[-1], trials)

        return trials

    def observe(self, trials):
        """Observe the `trials` new state of result.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        super().observe(trials)
        for trial in trials:
            if trial.status == "broken":
                trial = self.strategy.infer(trial)
            if trial.objective is not None:
                self.trial_results[self.get_id(trial)] = trial.objective.value
            runs = self.trial_meta.get(self.get_id(trial), [])
            for run in runs:
                job = FakeJob(run, trial)
                self.iterations[job.id[0]].register_result(job)
                self.bohb.new_result(job)

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        return self.iteration == len(self.budgets) and all(
            it.is_finished for it in self.iterations
        )
