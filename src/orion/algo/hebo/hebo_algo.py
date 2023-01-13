"""
:mod:`orion.algo.HEBO.hebo -- Orion adapter for the HEBO algorithm.
============================================

The HEBO algorithm implementation can be found at https://github.com/huawei-noah/HEBO
"""
from __future__ import annotations

import copy
import typing
import warnings
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from typing_extensions import Literal, TypedDict  # type: ignore

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Dimension, Fidelity, Space
from orion.core.utils.format_trials import dict_to_trial
from orion.core.utils.module_import import ImportOptional
from orion.core.utils.random_state import RandomState, control_randomness
from orion.core.worker.transformer import TransformedDimension
from orion.core.worker.trial import Trial

with ImportOptional("HEBO") as import_optional:
    import hebo
    from hebo.acquisitions.acq import MACE, Acquisition
    from hebo.design_space import DesignSpace
    from hebo.design_space.param import Parameter
    from torch.quasirandom import SobolEngine

if import_optional.failed:
    MACE = object  # noqa: F811

if typing.TYPE_CHECKING and import_optional.failed:
    Acquisition = object  # noqa
    DesignSpace = object  # noqa
    Parameter = object  # noqa
    SobolEngine = object  # noqa

logger = get_logger(__name__)

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
properly_seeded_models: set[ModelName] = {"gp", "gpy", "gpy_mlp", "rf", "catboost"}
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


class HeboModelState(TypedDict):
    """Typed dict for the state of the HEBO class from `hebo.optimizers.hebo`."""

    space: DesignSpace
    # pylint: disable=invalid-name
    es: str
    X: pd.DataFrame
    y: np.ndarray
    model_name: str
    rand_sample: int
    sobol: SobolEngine
    acq_cls: type[Acquisition]
    _model_config: dict | None


class HEBO(BaseAlgorithm):
    """Adapter for the HEBO algorithm from https://github.com/huawei-noah/HEBO

    Parameters
    ----------
    :param space: Optimisation space with priors for each dimension.
    :param seed: Base seed for the random number generators. Defaults to `None`, in which case the
    randomness is not seeded.
    :param parameters: Parameters for the HEBO algorithm.
    """

    requires_type: ClassVar[str | None] = None
    requires_shape: ClassVar[str | None] = "flattened"
    requires_dist: ClassVar[str | None] = None

    @dataclass(frozen=True)
    class Parameters:
        """Parameters of the HEBO algorithm."""

        model_name: ModelName = "gpy"
        """ Name of the model to use. See `ModelName` for the available values. """

        random_samples: int | None = None
        """ Number of random samples to suggest before optimization begins.
        If `None`, the number of dimensions in the space is used as the default. Otherwise, the max
        of the value and `2` is used.
        """

        acquisition_class: type[Acquisition] = MACE
        """ Acquisition class to use. """

        evolutionary_strategy: EvolutionStrategyName = "nsga2"
        """ Name of the evolutionary strategy to use. See `EvolutionStrategyName` for the list of
        possible values.
        """

        model_config: dict | None = None
        """ Keyword argument to be passed to the constructor of the model class that is selected
        with `model_name`.
        """

    def __init__(
        self,
        space: Space,
        seed: int | None = None,
        parameters: Parameters | dict | None = None,
    ):
        import_optional.ensure()

        super().__init__(space)
        if isinstance(parameters, dict):
            parameters = self.Parameters(**parameters)
        self.parameters: HEBO.Parameters = parameters or self.Parameters()

        self.seed = seed
        self.random_state: RandomState | None = None

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
        # NOTE: Need to seed the randomness here, since creating the model affects the global torch
        # RNG state. This way, we always get the same model for the same seed.
        if self.seed is not None:
            self.seed_rng(self.seed)

        self.hebo_space: DesignSpace = orion_space_to_hebo_space(self.space)

        with control_randomness(self):
            self.model = hebo.optimizers.hebo.HEBO(
                space=self.hebo_space,
                model_name=self.parameters.model_name,
                rand_sample=self.parameters.random_samples,
                acq_cls=self.parameters.acquisition_class,
                es=self.parameters.evolutionary_strategy,
                model_config=self.parameters.model_config,
            )

    def seed_rng(self, seed: int | None) -> None:
        """Seed the random number generators."""
        logger.debug("Using a base seed of %s.", seed)
        self.random_state = RandomState.seed(seed)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        base_state_dict = super().state_dict
        model_state = HeboModelState(
            space=self.model.space,
            X=self.model.X,
            y=self.model.y,
            es=self.model.es,
            model_name=self.model.model_name,
            acq_cls=self.model.acq_cls,
            rand_sample=self.model.rand_sample,
            _model_config=self.model._model_config,  # pylint:disable=protected-access
            sobol=self.model.sobol,
        )
        return copy.deepcopy(
            dict(
                **base_state_dict,
                model=model_state,
                random_state=self.random_state or RandomState.current(),
                parameters=self.parameters,
            )
        )

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super().set_state(state_dict)
        self.random_state = state_dict["random_state"]
        self.parameters = state_dict["parameters"]
        model_state = state_dict["model"]
        # NOTE: For now assuming that we can just store anything into the state dict
        for key, value in model_state.items():

            if not hasattr(self.model, key):
                raise RuntimeError(
                    f"The state dict has attribute {key} that is not in the model!"
                )
            setattr(self.model, key, value)

    def suggest(self, num: int) -> list[Trial]:
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
        trials: list[Trial] = []
        with control_randomness(self):
            v: pd.DataFrame = self.model.suggest(n_suggestions=num)
        point_dicts: dict[int, dict] = v.to_dict(orient="index")  # type: ignore

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

    def observe(self, trials: list[Trial]) -> None:
        """Observe the `trials` new state of result.

        Parameters
        ----------
        :param trials: New trials with their objectives.
        """
        new_xs: list[dict] = []
        new_ys: list[float] = []
        assert len(self.model.X) == self.n_observed

        for trial in trials:
            if not self.has_observed(trial):
                self.register(trial)
                new_x = trial.params
                if trial.objective is None:
                    # Trial is broken: ignore it.
                    continue
                new_y = trial.objective.value

                new_x = self._orion_params_to_hebo_params(new_x)
                new_xs.append(new_x)
                new_ys.append(new_y)

        x_df = pd.DataFrame(new_xs)
        y_array = np.array(new_ys).reshape([-1, 1])
        with control_randomness(self):
            self.model.observe(X=x_df, y=y_array)

    def _hebo_params_to_orion_params(self, hebo_params: dict) -> dict:
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

    def _orion_params_to_hebo_params(self, orion_params: dict) -> dict:
        """Fix any issues with the trials from Orion so they fit the `self.hebo_space`."""
        assert self.hebo_space is not None

        # NOTE: Remove the extra stuff (e.g. Fidelity dimension), before passing it to the
        # model. It's just a precaution, since the Hebo model probably fetches data using
        # the keys of its space, which doesn't have the Fidelity dimension anyway.

        params = {}
        for name, value in orion_params.items():
            orion_dim: Dimension = self.space[name]
            if orion_dim.type == "fidelity":
                continue
            from hebo.design_space.categorical_param import CategoricalPara

            hebo_dim: Parameter = self.hebo_space.paras[name]

            if isinstance(hebo_dim, CategoricalPara):
                if (
                    value not in hebo_dim.categories
                    and str(value) in hebo_dim.categories
                ):
                    value = str(value)
                assert value in hebo_dim.categories, (value, hebo_dim.categories)

            params[name] = value

        return params

    def _params_to_trial(self, orion_params: dict) -> Trial:
        """Create a Trial from a dict of hyper-parameters."""
        # Need to convert the {name: value} of point_dict into this format for Orion's Trial.
        # Add the max value for the Fidelity dimensions, if any.
        if self.fidelity_index is not None:
            fidelity_dim = self.space[self.fidelity_index]
            while isinstance(fidelity_dim, TransformedDimension):
                fidelity_dim = fidelity_dim.original_dimension
            assert isinstance(fidelity_dim, Fidelity)
            orion_params[self.fidelity_index] = float(fidelity_dim.high)
        trial: Trial = dict_to_trial(orion_params, space=self.space)
        return trial


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
        spec: dict[str, Any] = {"name": name}
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
