"""
:mod:`orion.algo.axoptimizer` -- Ax Wrapper
===========================================
"""
import contextlib
import copy
from typing import List, Optional

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Fidelity, Space
from orion.core.utils import format_trials
from orion.core.utils.flatten import flatten
from orion.core.utils.module_import import ImportOptional
from orion.core.worker.transformer import TransformedDimension

with ImportOptional("Ax") as import_optional:
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties

if import_optional.failed:
    # pylint: disable=invalid-name
    AxClient = None  # noqa: F811
    ObjectiveProperties = None  # noqa: F811


class AxOptimizer(BaseAlgorithm):
    """Wrapper around the `Ax platform <https://ax.dev/>`_ for multi-objectives
    optimization and constraints.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int, optional
        random seed for reproducibility. Works only for Sobol quasi-random
        generator and for BoTorch-powered models. For the latter models, the
        trials generated from the same optimization setup with the same seed,
        will be mostly similar, but the exact parameter values may still vary
        and trials latter in the optimizations will diverge more and more. This
        is because a degree of randomness is essential for high performance of
        the Bayesian optimization models and is not controlled by the seed.

        .. note:: In multi-threaded environments, the random seed is
           thread-safe, but does not actually guarantee reproducibility. Whether
           the outcomes will be exactly the same for two same operations that
           use the random seed, depends on whether the threads modify the random
           state in the same order across the two operations.  Default: ``None``
    n_initial_trials: int, optional
        Specific number of initialization trials.
        Initialization trials are generated quasi-randomly using Sobol.
    extra_objectives: sequence of str, optional
        List of metrics' name which are also objectives to minimize.
        threshold: The bound in the objective's threshold constraint.

        .. note:: Orion expects the `extra_objectives` results to be stored in
           `orion.core.worker.Trial.statistics`
    constraints: sequence of str, optional
        Dict of list of string representation of metrics constraints of form
        ["metric_name >= bound"], like ["m1 <= 3"]

        .. note:: Orion expects the `constraints` results to be stored in
           `orion.core.worker.Trial.constraints`
        .. note:: See https://ax.dev/docs/core.html#optimization-config for more
           details about how Ax expects its outcome constraints
    """

    requires_type = "numerical"
    requires_dist = None
    requires_shape = "flattened"

    def __init__(
        self,
        space: Space,
        seed: Optional[int] = None,
        n_initial_trials: Optional[int] = 20,
        extra_objectives: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
    ):
        import_optional.ensure()

        extra_objectives = set(extra_objectives if extra_objectives else [])
        constraints = constraints if constraints else []

        # Ax needs its max_trials property to always be set to build its client
        self.max_trials = None
        self._client_state = None
        self._trials_map = {}  # tmp

        super().__init__(
            space,
            seed=seed,
            n_initial_trials=n_initial_trials,
            extra_objectives=extra_objectives,
            constraints=constraints,
        )

        # test Ax parameters
        with self.get_client():
            pass
        self._client_state = None

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int
            Integer seed for the random number generator.

        .. note:: Ax does not promise deterministic trials generation and only
           similar trials generation.
        """
        if self._client_state is not None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not support reseeding an instantiated client"
            )
        self.seed = seed

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = copy.deepcopy(super().state_dict)
        # NOTE: AxClient.to_json_snapshot() seams to be more like the current internal
        #       state of the client than an independent snapshot
        state_dict["_client_state"] = copy.deepcopy(self._client_state)
        state_dict["_trials_map"] = copy.deepcopy(self._trials_map)

        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        Parameters
        ----------
        state_dict: dict
            Dictionary representing state of an algorithm
        """
        super().set_state(copy.deepcopy(state_dict))
        self._client_state = copy.deepcopy(state_dict.get("_client_state"))
        self._trials_map = copy.deepcopy(state_dict.get("_trials_map"))

    @contextlib.contextmanager
    def get_client(self):
        """Instantiate a new AxClient from previous snapshot"""
        if self._client_state is not None:
            # Copy client state because `from_json_snapshot` modifies it...
            client = AxClient.from_json_snapshot(copy.deepcopy(self._client_state))
        else:
            client = AxClient(
                random_seed=self.seed,
                enforce_sequential_optimization=False,
                verbose_logging=False,
            )

            client.create_experiment(
                parameters=orion_space_to_axoptimizer_space(self.space),
                choose_generation_strategy_kwargs={
                    "num_initialization_trials": self.n_initial_trials,
                    "max_parallelism_override": self.max_trials,
                },
                objectives={
                    "objective": ObjectiveProperties(minimize=True),
                    **{
                        o: ObjectiveProperties(minimize=True)
                        for o in self.extra_objectives
                    },
                },
                outcome_constraints=self.constraints,
            )

        yield client

        self._client_state = client.to_json_snapshot()

    def suggest(self, num):
        """Suggest a number of new sets of parameters.

        Parameters
        ----------
        num: int
            Number of trials to suggest. The algorithm may return less than the number of trials
            requested.

        Returns
        -------
        list of trials
            A list of trials representing values suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.


        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """
        trials = []
        with self.get_client() as client:
            _trials, _ = client.get_next_trials(num)
            for trial_index, parameters in _trials.items():
                parameters = AxOptimizer.reverse_params(parameters, self.space)

                # Ax does not support Fidelity dimension type so fake it with
                # its max
                if self.fidelity_index is not None:
                    # Convert 0-dim arrays into python numbers so their type can
                    # be validated by Ax
                    fidelity_dim = self.space[self.fidelity_index]
                    while isinstance(fidelity_dim, TransformedDimension):
                        fidelity_dim = fidelity_dim.original_dimension
                    assert isinstance(fidelity_dim, Fidelity)
                    parameters[self.fidelity_index] = float(fidelity_dim.high)

                new_trial = format_trials.dict_to_trial(parameters, self.space)

                if not self.has_suggested(new_trial):
                    self.register(new_trial)
                    trials.append(new_trial)
                    self._trials_map[self.get_id(new_trial)] = trial_index  # tmp

        return trials

    def observe(self, trials):
        """Observe the `trials` new state of result.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        with self.get_client() as ax_client:
            for trial in trials:
                if not self.has_suggested(trial):
                    _, trial_index = ax_client.attach_trial(
                        AxOptimizer.transform_params(flatten(trial.params), self.space)
                    )
                    self._trials_map[self.get_id(trial)] = trial_index

                if not self.has_observed(trial):
                    # Check the trial status
                    trial_status = trial.status

                    # If the trial status is `completed`
                    if trial_status == "completed":
                        # Complete it in Ax
                        ax_trial_index = self._trials_map[self.get_id(trial)]
                        raw_data = {
                            "objective": trial.objective.value,
                            **{
                                s.name: s.value
                                for s in trial.statistics
                                if s.name in self.extra_objectives
                            },
                            **{r.name: r.value for r in trial.constraints},
                        }
                        ax_client.complete_trial(
                            trial_index=ax_trial_index, raw_data=raw_data
                        )

                    # If the trial status is `broken`
                    elif trial_status == "broken":
                        # Set is as broken is Ax
                        ax_trial_index = self._trials_map[self.get_id(trial)]
                        ax_client.log_trial_failure(ax_trial_index)

                    # Register the unobserved trial
                    self.register(trial)

    @classmethod
    def transform_params(cls, orion_params, space):
        """Convert orion parameter values"""
        ax_params = {}
        for dim in space.values():
            if dim.type == "fidelity":
                continue

            ax_params[dim.name] = orion_params[dim.name]

        return ax_params

    @classmethod
    def reverse_params(cls, ax_params, space):  # pylint: disable=unused-argument
        """Reverse converted `choices` dimensions values to their original types"""
        orion_params = copy.deepcopy(ax_params)
        return orion_params


# pylint: disable=invalid-name
def orion_space_to_axoptimizer_space(orion_space):
    """Convert Orion's definition of problem's domain to an axoptimizer
    compatible one."""
    axoptimizer_space = []

    for dimension in orion_space.values():
        if dimension.type == "fidelity":
            continue

        dimensions_dict = {}
        dimensions_dict["name"] = dimension.name

        dimensions_dict["type"] = "range"
        # Convert 0-dim arrays into python numbers so their type can be
        # validated by Ax
        dimensions_dict["bounds"] = [float(i) for i in dimension.interval()]

        if dimension.prior_name in "uniform":
            dimensions_dict["value_type"] = "float"
            dimensions_dict["log_scale"] = False

        elif dimension.prior_name == "reciprocal":
            dimensions_dict["value_type"] = "float"
            dimensions_dict["log_scale"] = True

        elif dimension.prior_name in ["int_uniform", "choices"]:
            dimensions_dict["value_type"] = "int"
            dimensions_dict["log_scale"] = False

        elif dimension.prior_name == "int_reciprocal":
            dimensions_dict["value_type"] = "int"
            dimensions_dict["log_scale"] = True

        else:
            raise TypeError(
                "Ax only supports `uniform`, `reciprocal`, `int_uniform`, "
                f"`int_reciprocal` and `choices`: `{dimension.prior_name}`"
            )

        axoptimizer_space.append(dimensions_dict)

    return axoptimizer_space
