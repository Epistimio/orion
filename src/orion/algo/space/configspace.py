"""
Configspace Conversion
======================
"""
from __future__ import annotations

from copy import deepcopy
from functools import singledispatch
from math import log10

from orion.algo.space import (
    Categorical,
    Dimension,
    Fidelity,
    Integer,
    Real,
    Space,
    SpaceConverter,
    to_orionspace,
)

try:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        FloatHyperparameter,
        Hyperparameter,
        IntegerHyperparameter,
        NormalFloatHyperparameter,
        NormalIntegerHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    IMPORT_ERROR = None

except ImportError as err:
    IMPORT_ERROR = err

    # pylint: disable=too-few-public-methods
    class DummyType:
        """Dummy type for type hints"""

    IntegerHyperparameter = DummyType
    FloatHyperparameter = DummyType
    ConfigurationSpace = DummyType
    Hyperparameter = DummyType
    UniformFloatHyperparameter = DummyType
    NormalFloatHyperparameter = DummyType
    UniformIntegerHyperparameter = DummyType
    NormalIntegerHyperparameter = DummyType
    CategoricalHyperparameter = DummyType


class UnsupportedPrior(Exception):
    """Raised when the converting an unsupported dimension to configspace"""


def _qantization(dim: Dimension) -> float:
    """Convert precision to the quantization factor"""
    if dim.precision:
        return 10 ** (-dim.precision)
    return None


def _upsert(array, i, value):
    cp = len(array) - i

    if value is None:
        return

    if cp == 0:
        array.append(value)
        return

    if cp > 0:
        array[i] = value
        return

    raise IndexError()


# pylint: disable=protected-access,unused-argument
def normalize_args(dim: Dimension, rv, kwarg_order=None) -> dict:
    """Create an argument array from kwargs"""
    if kwarg_order is None:
        kwarg_order = ["loc", "scale"]

    if len(dim._kwargs) == 0:
        return dim._args[: len(kwarg_order)]

    args = list(deepcopy(dim._args))

    for i, kw in enumerate(kwarg_order):
        _upsert(args, i, dim._kwargs.get(kw))

    return args[: len(kwarg_order)]


class ToConfigSpace(SpaceConverter[Hyperparameter]):
    """Convert an Orion space into a configspace"""

    def __init__(self) -> None:
        if IMPORT_ERROR is not None:
            raise IMPORT_ERROR

    def dimension(self, dim: Dimension) -> None:
        """Raise an error if the visitor is called on an abstract class"""
        raise NotImplementedError()

    def real(self, dim: Real) -> FloatHyperparameter:
        """Convert a real dimension into a configspace equivalent"""
        if dim.prior_name in ("reciprocal", "uniform"):
            lower, upper = dim.interval()

            return UniformFloatHyperparameter(
                name=dim.name,
                lower=lower,
                upper=upper,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=dim.prior_name == "reciprocal",
            )

        if dim.prior_name in ("normal", "norm"):
            a, b = normalize_args(dim, dim.prior)

            kwargs = dict(
                name=dim.name,
                mu=a,
                sigma=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=False,
                lower=dim.low if hasattr(dim, "low") else None,
                upper=dim.high if hasattr(dim, "high") else None,
            )

            return NormalFloatHyperparameter(**kwargs)

        raise UnsupportedPrior(f'Prior "{dim.prior_name}" is not supported')

    def integer(self, dim: Integer) -> IntegerHyperparameter:
        """Convert a integer dimension into a configspace equivalent"""

        if dim.prior_name in ("int_uniform", "int_reciprocal"):
            lower, upper = dim.interval()

            return UniformIntegerHyperparameter(
                name=dim.name,
                lower=lower,
                upper=upper,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=dim.prior_name == "int_reciprocal",
            )

        if dim.prior_name in ("int_norm", "normal"):
            a, b = normalize_args(dim, dim.prior)

            kwargs = dict(
                name=dim.name,
                mu=a,
                sigma=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=False,
                lower=dim.low if hasattr(dim, "low") else None,
                upper=dim.high if hasattr(dim, "high") else None,
            )

            return NormalIntegerHyperparameter(**kwargs)

        raise UnsupportedPrior(f'Prior "{dim.prior_name}" is not supported')

    def categorical(self, dim: Categorical) -> CategoricalHyperparameter:
        """Convert a categorical dimension into a configspace equivalent"""
        return CategoricalHyperparameter(
            name=dim.name,
            choices=dim.categories,
            weights=dim._probs,
        )

    # pylint: disable=unused-argument
    def fidelity(self, dim: Fidelity) -> None:
        """Ignores fidelity dimension as configspace does not have an equivalent"""
        return None

    def space(self, space: Space) -> ConfigurationSpace:
        """Convert orion space to configspace"""
        cspace = ConfigurationSpace()
        dims = []

        for _, dim in space.items():
            cdim = self.convert_dimension(dim)

            if cdim:
                dims.append(cdim)

        cspace.add_hyperparameters(dims)
        return cspace


def to_configspace(space: Space) -> ConfigurationSpace:
    """Convert orion space to configspace

    Notes
    -----
    ``ConfigurationSpace`` will set its own default values
    if not set inside ``Space``

    """
    conversion = ToConfigSpace()
    return conversion.space(space)


@singledispatch
def to_oriondim(dim: Hyperparameter) -> Dimension:
    """Convert a config space hyperparameter to an orion dimension"""
    raise NotImplementedError(f"Dimension {dim} is not supported by Orion")


@to_oriondim.register
def _from_categorical(dim: CategoricalHyperparameter) -> Categorical:
    """Builds a categorical dimension from a categorical hyperparameter"""
    choices = dict(zip(dim.choices, dim.probabilities))
    return Categorical(dim.name, choices)


@to_oriondim.register(UniformIntegerHyperparameter)
@to_oriondim.register(UniformFloatHyperparameter)
def _from_uniform(dim: Hyperparameter) -> Integer | Real:
    """Builds a uniform dimension from a uniform hyperparameter"""

    klass = Integer
    args = []
    kwargs = dict(
        # NOTE: Config space always has a config value
        # so orion-space would get it as well
        default_value=dim.default_value
    )

    if isinstance(dim, UniformFloatHyperparameter):
        klass = Real
    else:
        kwargs["precision"] = int(-log10(dim.q)) if dim.q else 4

    if dim.log:
        dist = "reciprocal"
        args.append(dim.lower)
        args.append(dim.upper)
    else:
        # NB: scipy uniform [loc, scale], configspace [min, max] with max = loc + scale, loc = min
        loc = dim.lower
        scale = dim.upper - dim.lower

        dist = "uniform"
        args.append(loc)
        args.append(scale)

    return klass(dim.name, dist, *args, **kwargs)


@to_oriondim.register(NormalFloatHyperparameter)
@to_oriondim.register(NormalIntegerHyperparameter)
def _from_normal(dim: Hyperparameter) -> Integer | Real:
    """Builds a normal dimension from a normal hyperparameter"""

    klass = Integer
    args = []
    kwargs = dict(
        # NOTE: Config space always has a config value
        # so orion-space would get it as well
        default_value=dim.default_value
    )

    if isinstance(dim, NormalFloatHyperparameter):
        klass = Real
    else:
        kwargs["precision"] = int(-log10(dim.q)) if dim.q else 4

    dist = "norm"
    args.append(dim.mu)
    args.append(dim.sigma)

    if dim.lower:
        kwargs["low"] = dim.lower
        kwargs["high"] = dim.upper

    return klass(dim.name, dist, *args, **kwargs)


@to_orionspace.register
def configspace_to_orionspace(cspace: ConfigurationSpace) -> Space:
    """Convert from orion space to configspace

    Notes
    -----
    ``ConfigurationSpace`` will set default values for each dimensions of ``Space``

    """
    space = Space()

    for cdim in cspace.get_hyperparameters_dict().values():
        odim = to_oriondim(cdim)
        space.register(odim)

    return space
