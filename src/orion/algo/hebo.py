""" Utility class for saving / restoring state of the global random number generators. """
from __future__ import annotations

import contextlib
import copy
import random
import warnings
from dataclasses import dataclass, replace
from logging import getLogger as get_logger
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from hebo.acquisitions.acq import MACE, Acquisition
from hebo.design_space import DesignSpace
from hebo.design_space.param import Parameter
from hebo.optimizers.hebo import HEBO as Hebo
from torch.quasirandom import SobolEngine

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Dimension, Fidelity, Space
from orion.core.utils.format_trials import dict_to_trial
from orion.core.worker.trial import Trial

try:
    from typing import Literal, TypedDict  # type: ignore
except ImportError:
    from typing_extensions import Literal, TypedDict  # type: ignore

properly_seeded_models = {"gp", "gpy", "gpy_mlp", "rf", "catboost"}
logger = get_logger(__name__)


@dataclass(frozen=True)
class RandomState:
    """Immutable dataclass that holds the state of the various random number generators."""

    random_state: Any
    """ RNG state for the `random` module. """

    numpy_rng_state: Dict[str, Any]
    """ RNG state for the `numpy.random` module. """

    torch_rng_state: torch.Tensor
    """ RNG state for the `torch` module. """

    torch_cuda_rng_state: List[torch.Tensor]
    """ RNG states for the `torch.cuda` module, for each cuda device available. """

    base_seed: Optional[int] = None
    """Base seed that was used to create this object (the `seed` argument to `RandomState.seed`)."""

    @classmethod
    def current(cls) -> "RandomState":
        """Returns the current random state.

        NOTE: The `base_seed` of the returned object will be None.
        """
        return cls(
            random_state=random.getstate(),
            numpy_rng_state=np.random.get_state(),
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state_all(),
        )

    def set(self) -> None:
        """Sets the random state using the values in `self`."""
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_rng_state)
        torch.random.set_rng_state(self.torch_rng_state)
        torch.cuda.set_rng_state_all(self.torch_cuda_rng_state)

    @classmethod
    def seed(cls, base_seed: Optional[int]) -> "RandomState":
        """Seeds all the RNGs using the given seed, and returns the resulting RandomState object."""
        random.seed(base_seed)
        numpy_seed = random.randint(0, 2**32 - 1)
        torch_seed = random.randint(0, 2**32 - 1)
        np.random.seed(numpy_seed)
        torch.random.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)

        random_state = cls.current()
        if base_seed is not None:
            # Add the base_seed property in this case.
            random_state = replace(random_state, base_seed=base_seed)
        return random_state


ModelName = Literal[
    "gp",
    "gpy",
    "gpy_mlp",
    "rf",
    "deep_ensemble",
    "masked_deep_ensemble",
    "fe_deep_ensemble",
    "gumbel",
    "catboost",
]
EvolutionStrategyName = Literal[
    "ga",
    "brkga",
    "de",
    "nelder-mead",
    "pattern-search",
    "cmaes",
    "pso",
    "nsga2",
    "rnsga2",
    "nsga3",
    "unsga3",
    "rnsga3",
    "moead",
    "ctaea",
]


class BaseAlgorithmStateDict(TypedDict):
    """State dict of the `BaseAlgorithm` class.
    NOTE: This is meant to immitate an inner class of `BaseAlgorithm`.
    """

    _trials_info: Dict[str, Tuple[Trial, Trial.Result]]


class HeboModelState(TypedDict):
    """Typed dict for the sate of the HEBO class from `hebo.optimizers.hebo`."""

    space: DesignSpace
    es: str  # pylint:disable=invalid-name
    X: pd.DataFrame  # pylint:disable=invalid-name
    y: np.ndarray  # pylint:disable=invalid-name
    model_name: str
    rand_sample: int
    sobol: SobolEngine
    acq_cls: Type[Acquisition]
    _model_config: Optional[Dict]


class HEBO(BaseAlgorithm):
    """Adapter for the HEBO algorithm from https://github.com/huawei-noah/HEBO
    Parameters
    ----------
    :param space: Optimisation space with priors for each dimension.
    :seed seed: Base seed for the random number generators. Defaults to `None`, in which case the
    randomness is not seeded.
    """

    requires_type: ClassVar[Optional[str]] = None
    requires_shape: ClassVar[Optional[str]] = "flattened"
    requires_dist: ClassVar[Optional[str]] = None

    @dataclass(frozen=True)
    class Parameters:
        """Parameters of the HEBO algorithm."""

        model_name: ModelName = "gpy"
        """ Name of the model to use. See `ModelName` for the available values. """

        random_samples: Optional[int] = None
        """ Number of random samples to suggest before optimization begins.
        If `None`, the number of dimensions in the space is used as the default. Otherwise, the max
        of the value and `2` is used.
        """

        acquisition_class: Type[Acquisition] = MACE
        """ Acquisition class to use. """

        evolutionary_strategy: EvolutionStrategyName = "nsga2"
        """ Name of the evolutionary strategy to use. See `EvolutionStrategyName` for the list of
        possible values.
        """

        model_config: Optional[Dict] = None
        """ Keyword argument to be passed to the constructor of the model class that is selected
        with `model_name`.
        """

    class StateDict(BaseAlgorithmStateDict):
        """TypedDict for the state of this algorithm."""

        model: Optional[HeboModelState]
        """ State dict of the model. """

        random_state: RandomState
        """ State of the random number generators. """

        parameters: HEBO.Parameters
        """ Hyper-parameters of the HEBO algorithm. """

    def __init__(
        self,
        space: Space,
        seed: int = None,
        parameters: HEBO.Parameters | Dict | None = None,
    ):
        if isinstance(parameters, dict):
            parameters = self.Parameters(**parameters)
        parameters = parameters or self.Parameters()
        super().__init__(space, seed=seed, parameters=parameters)
        self.seed: Optional[int] = seed
        self.parameters: HEBO.Parameters = parameters
        self.random_state: Optional[RandomState] = None
        self.hebo_space: Optional[DesignSpace] = None
        self.model: Optional[Hebo] = None
        if (
            self.parameters.model_name not in properly_seeded_models
            and seed is not None
        ):
            warnings.warn(
                UserWarning(
                    f"The randomness used by the chosen model '{self.parameters.model_name}' "
                    f"cannot be properly seeded. The model will still work, but the results may "
                    f"not be reproducible, and the random state will not be properly "
                    f"saved/restored during checkpointing."
                )
            )

    def _initialize(self):
        """Seed the randomness and create the model.
        TODO: Currently the `space` property is changed after the algo is initialized. Once that is
        fixed/cleaned-up, make the `space` read-only, and prevent multiple calls to `_initialize`.
        """
        # NOTE: Need to seed the randomness here, rather than only once in the constructor, since
        # creating the model affects the global torch RNG state. This way, we always get the same
        # model for the same seed, even if the space is changed many times.
        if self.seed is not None:
            self.seed_rng(self.seed)
        self.hebo_space = orion_space_to_hebo_space(self.space)
        with self._control_randomness():
            self.model = Hebo(
                space=self.hebo_space,
                model_name=self.parameters.model_name,
                rand_sample=self.parameters.random_samples,
                acq_cls=self.parameters.acquisition_class,
                es=self.parameters.evolutionary_strategy,
                model_config=self.parameters.model_config,
            )

    @property
    def space(self) -> Space:
        """Space of hyper-parameters that this algorithm is optimizing."""
        return self._space

    @space.setter
    def space(self, value: Space) -> None:
        """Sets the HPO space.
        When setting a new value, this has the side-effect of seeding the random state and
        initializing the model.
        NOTE: This property is actually updated *after* the algorithm is created, when the space is
        transformed.
        """
        space_changed = self.space != value
        self._space = value
        if space_changed:
            self._initialize()

    def seed_rng(self, seed: Optional[int]) -> None:
        """Seed the random number generators."""
        logger.debug("Using a base seed of %s.", seed)
        self.random_state = RandomState.seed(seed)

    @property
    def state_dict(self) -> HEBO.StateDict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        base_state_dict: BaseAlgorithmStateDict = super().state_dict  # type: ignore
        model_state: Optional[HeboModelState] = None
        if self.model is not None:
            acquisition_class = self.model.acq_cls
            assert issubclass(acquisition_class, Acquisition)
            model_state = HeboModelState(
                space=self.model.space,
                X=self.model.X,
                y=self.model.y,
                es=self.model.es,
                model_name=self.model.model_name,
                acq_cls=acquisition_class,
                rand_sample=self.model.rand_sample,
                _model_config=self.model._model_config,  # pylint:disable=protected-access
                sobol=self.model.sobol,
            )
        state_dict: HEBO.StateDict = self.StateDict(
            _trials_info=base_state_dict["_trials_info"],
            model=model_state,
            random_state=self.random_state or RandomState.current(),
            parameters=self.parameters,
        )
        return copy.deepcopy(state_dict)

    def set_state(self, state_dict: HEBO.StateDict) -> None:
        """Reset the state of the algorithm based on the given state_dict
        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)
        self.random_state = state_dict["random_state"]
        self.parameters = state_dict["parameters"]
        model_state = state_dict["model"]
        if model_state is not None:
            if self.model is None:
                self._initialize()
            assert self.model is not None
            # NOTE: For now assuming that we can just store anything into the state dict
            for key, value in model_state.items():
                assert hasattr(self.model, key)
                setattr(self.model, key, value)

    def suggest(self, num: int) -> List[Trial]:
        """Suggest `num` new sets of hyper-parameters to try.
        Parameters
        ----------
        num: int
            Number of trials to suggest. The algorithm may return less than the number of
            trials requested.
        Returns
        -------
        A list of trials representing values suggested by the algorithm.
        """
        if self.model is None:
            self._initialize()
        if self.model is None:
            raise RuntimeError("Model did not initialize properly")

        trials: List[Trial] = []
        with self._control_randomness():
            v: pd.DataFrame = self.model.suggest(n_suggestions=num)
        point_dicts: Dict[int, Dict] = v.to_dict(orient="index")  # type: ignore

        for point_index, params_dict in point_dicts.items():
            if self.is_done:
                break
            params_dict = self._hebo_params_to_orion_params(params_dict)
            new_trial = self._params_to_trial(params_dict)

            if not self.has_suggested(new_trial):
                self.register(new_trial)
                trials.append(new_trial)
                logger.debug("Suggestion %s: %s", point_index, new_trial)
        return trials

    def observe(self, trials: List[Trial]) -> None:
        """Observe the `trials` new state of result.
        Parameters
        ----------
        :param trials: New trials with their objectives.
        """
        if self.model is None:
            self._initialize()
        assert self.model is not None
        new_xs: List[Dict] = []
        new_ys: List[float] = []
        assert len(self.model.X) == self.n_observed

        for trial in trials:
            trial = self.format_trial(trial)

            if not self.has_observed(trial):
                self.register(trial)
                new_x = trial.params
                new_y = trial.results[0].value

                new_x = self._orion_params_to_hebo_params(new_x)
                new_xs.append(new_x)
                new_ys.append(new_y)

        x_df = pd.DataFrame(new_xs)
        y_array = np.array(new_ys).reshape([-1, 1])
        with self._control_randomness():
            self.model.observe(X=x_df, y=y_array)

    def _hebo_params_to_orion_params(self, hebo_params: Dict) -> Dict:
        """Fix any issues with the outputs of the HEBO algo so they fit `self.space`."""
        orion_params = {}
        for name, value in hebo_params.items():
            dim: Dimension = self.space[name]
            from orion.core.worker.transformer import ReshapedDimension

            if (
                dim.type == "categorical"
                and dim.prior_name == "choices"
                and value not in dim
            ):
                potential_vals = [v for v in dim.interval() if str(v) == value]
                if len(potential_vals) == 1:
                    value = potential_vals[0]
                else:
                    raise RuntimeError(
                        f"Value {value} is not contained in the dimension {dim}, and "
                        f"{len(potential_vals)} could match it."
                    )
            elif isinstance(dim, ReshapedDimension):
                # BUG: https://github.com/Epistimio/orion/issues/800
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # note: need to make sure `value` isn't a bool, since issubclass(bool, int).
                    value = np.array(value)
                # assert value in dim  # NOTE: Doesn't work! Raises an issue (index is None + int).
            orion_params[name] = value

        if self._params_to_trial(orion_params) not in self.space:
            raise RuntimeError(
                f"Unable to fix all the issues: params {orion_params} still isn't in space "
                f"{self.space}!"
            )

        return orion_params

    def _orion_params_to_hebo_params(self, orion_params: Dict) -> Dict:
        """Fix any issues with the trials from Orion so they fit the `self.hebo_space`."""
        assert self.hebo_space is not None

        # NOTE: Remove the extra stuff (e.g. Fidelity dimension), before passing it to the
        # model. It's just a precaution, since the Hebo model probably fetches data using
        # the keys of its space, which doesn't have the Fidelity dimension anyway.

        params = {}
        for name, value in orion_params.items():
            orion_dim: Dimension = self.space[name]
            hebo_dim: Parameter = self.hebo_space.paras[name]

            if orion_dim.type == "fidelity":
                continue
            from hebo.design_space.categorical_param import CategoricalPara

            if isinstance(hebo_dim, CategoricalPara):
                if (
                    value not in hebo_dim.categories
                    and str(value) in hebo_dim.categories
                ):
                    value = str(value)
                assert value in hebo_dim.categories, (value, hebo_dim.categories)

            params[name] = value

        return params

    def _params_to_trial(self, orion_params: Dict) -> Trial:
        """Create a Trial from a dict of hyper-parameters."""
        # Need to convert the {name: value} of point_dict into this format for Orion's Trial.
        # Add the max value for the Fidelity dimensions, if any.
        if self.fidelity_index is not None:
            fidelity_dim: Fidelity = self.space[self.fidelity_index]
            orion_params[self.fidelity_index] = fidelity_dim.high
        trial: Trial = dict_to_trial(orion_params, space=self.space)
        return trial

    @contextlib.contextmanager
    def _control_randomness(self):
        """Seeds the randomness inside the indented block of code using `self.random_state`.
        NOTE: This only has an effect if `seed_rng` was called previously, i.e. if
        `self.random_state` is not None.
        """
        if self.random_state is None:
            yield
            return

        # Save the initial random state.
        initial_rng_state = RandomState.current()
        # Set the random state.
        self.random_state.set()
        yield
        # Update the random state stored on `self`, so that the changes inside the block are
        # reflected in the RandomState object.
        self.random_state = RandomState.current()
        # Reset the initial state.
        initial_rng_state.set()


def orion_space_to_hebo_space(space: Space) -> DesignSpace:
    """Get the HEBO-equivalent space for the `Space` `space`.
    Parameters
    ----------
    :param space: `Space` instance.
    Returns
    -------
    a `DesignSpace` from the `hebo` package.
    Raises
    ------
    NotImplementedError
        If there is an unsupported dimension or prior type in `space`.
    """
    specs = []

    ds = DesignSpace()
    name: str
    dimension: Dimension
    for name, dimension in space.items():
        spec: Dict[str, Any] = {"name": name}
        prior_name = dimension.prior_name
        bounds = dimension.interval()
        if dimension.shape:
            raise NotImplementedError(
                f"HEBO algorithm doesn't support dimension {dimension} since it has a shape."
            )
        if dimension.type == "fidelity":
            # Ignore that dimension: Don't include it in the space for Hebo to optimize.
            continue

        # BUG: https://github.com/Epistimio/orion/issues/800
        bounds = tuple(b.item() if isinstance(b, np.ndarray) else b for b in bounds)

        if prior_name == "choices":
            categories = [str(b) for b in bounds]
            spec.update(type="cat", categories=categories)
        elif prior_name == "uniform":
            spec.update(type="num", lb=bounds[0], ub=bounds[1])
        elif prior_name == "reciprocal":
            spec.update(type="pow", lb=bounds[0], ub=bounds[1])
        elif prior_name == "int_uniform":
            spec.update(type="int", lb=bounds[0], ub=bounds[1])
        elif prior_name == "int_reciprocal":
            spec.update(type="pow_int", lb=bounds[0], ub=bounds[1])
        else:
            raise NotImplementedError(prior_name, dimension)
        specs.append(spec)
    ds.parse(specs)
    return ds
