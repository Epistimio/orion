from math import log10
from typing import Optional

from orion.algo.space import (
    Categorical,
    Dimension,
    Fidelity,
    Integer,
    Real,
    Space,
    Visitor,
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

    IntegerHyperparameter = object()
    FloatHyperparameter = object()
    ConfigurationSpace = object()
    Hyperparameter = object()
    UniformFloatHyperparameter = object()
    NormalFloatHyperparameter = object()
    UniformIntegerHyperparameter = object()
    NormalIntegerHyperparameter = object()
    CategoricalHyperparamete = object()


def _qantization(dim: Dimension) -> float:
    """Convert precision to the quantization factor"""
    if dim.precision:
        return 10 ** (-dim.precision)
    return None


class ToConfigSpace(Visitor[Optional[Hyperparameter]]):
    """Convert an Orion space into a configspace"""

    def __init__(self) -> None:
        if IMPORT_ERROR is not None:
            raise IMPORT_ERROR

    def dimension(self, dim: Dimension) -> None:
        """Raise an error if the visitor is called on an abstract class"""
        raise NotImplementedError()

    def real(self, dim: Real) -> Optional[FloatHyperparameter]:
        """Convert a real dimension into a configspace equivalent"""
        if dim.prior_name in ("reciprocal", "uniform"):
            a, b = dim._args

            return UniformFloatHyperparameter(
                name=dim.name,
                lower=a,
                upper=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=dim.prior_name == "reciprocal",
            )

        if dim.prior_name in ("normal", "norm"):
            a, b = dim._args

            return NormalFloatHyperparameter(
                name=dim.name,
                mu=a,
                sigma=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=False,
                lower=dim.low if hasattr(dim, "low") else None,
                upper=dim.high if hasattr(dim, "high") else None,
            )

        return

    def integer(self, dim: Integer) -> Optional[IntegerHyperparameter]:
        """Convert a integer dimension into a configspace equivalent"""
        if dim.prior_name in ("int_uniform", "int_reciprocal"):
            a, b = dim._args

            return UniformIntegerHyperparameter(
                name=dim.name,
                lower=a,
                upper=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=dim.prior_name == "int_reciprocal",
            )

        if dim.prior_name in ("norm", "normal"):
            a, b = dim._args

            return NormalIntegerHyperparameter(
                name=dim.name,
                mu=a,
                sigma=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=False,
                lower=dim.low if hasattr(dim, "low") else None,
                upper=dim.high if hasattr(dim, "high") else None,
            )

        return None

    def categorical(self, dim: Categorical) -> Optional[CategoricalHyperparameter]:
        """Convert a categorical dimension into a configspace equivalent"""
        return CategoricalHyperparameter(
            name=dim.name,
            choices=dim.categories,
            weights=dim._probs,
        )

    def fidelity(self, dim: Fidelity) -> None:
        """Ignores fidelity dimension as configspace does not have an equivalent"""
        return None

    def space(self, space: Space) -> ConfigurationSpace:
        """Convert orion space to configspace"""
        cspace = ConfigurationSpace()
        dims = []

        for _, dim in space.items():
            cdim = self.visit(dim)

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


def to_oriondim(dim: Hyperparameter) -> Dimension:
    """Convert a config space hyperparameter to an orion dimension"""

    if isinstance(dim, CategoricalHyperparameter):
        choices = {k: w for k, w in zip(dim.choices, dim.probabilities)}
        return Categorical(dim.name, choices)

    klass = Integer
    args = []
    kwargs = dict(
        # NOTE: Config space always has a config value
        # so orion-space would get it as well
        default_value=dim.default_value
    )

    if isinstance(dim, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
        if isinstance(dim, UniformFloatHyperparameter):
            klass = Real
        else:
            kwargs["precision"] = int(-log10(dim.q)) if dim.q else 4

        dist = "uniform"
        args.append(dim.lower)
        args.append(dim.upper)

        if dim.log:
            dist = "reciprocal"

    if isinstance(dim, (NormalFloatHyperparameter, NormalIntegerHyperparameter)):
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

    for _, cdim in cspace.get_hyperparameters_dict().items():
        odim = to_oriondim(cdim)
        space.register(odim)

    return space
