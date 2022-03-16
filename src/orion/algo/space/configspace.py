try:
    import ConfigSpace as cs
    import ConfigSpace.hyperparameters as csh

    IMPORT_ERROR = None
except ImportError as err:
    IMPORT_ERROR = err

from cmath import log10
from orion.algo.space import Visitor, Space, Real, Integer, Categorical


def _qantization(dim):
    """Convert precision to the quantization factor"""
    if dim.precision:
        return 10 ** (- dim.precision)
    return None


class ToConfigSpace(Visitor):
    """Convert an Orion space into a configspace"""

    def __init__(self) -> None:
        if IMPORT_ERROR is not None:
            raise IMPORT_ERROR

    def dimension(self, dim):
        """Raise an error if the visitor is called on an abstract class"""
        raise NotImplementedError()

    def real(self, dim):
        """Convert a real dimension into a configspace equivalent"""
        if dim.prior_name == ('loguniform', 'reciprocal', 'uniform'):
            a, b = dim._args

            return csh.UniformFloatHyperparameter(
                name=dim.name,
                lower=a,
                upper=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=dim.prior_name == 'reciprocal',
            )

        if dim.prior_name in ('normal', 'norm'):
            a, b = dim._args

            return csh.NormalFloatHyperparameter(
                name=dim.name,
                mu=a,
                sigma=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=False,
                lower=dim.low if hasattr(dim, 'low') else None,
                upper=dim.high if hasattr(dim, 'high') else None,
            )

        return

    def integer(self, dim):
        """Convert a integer dimension into a configspace equivalent"""
        if dim.prior_name == ('int_uniform', 'int_reciprocal'):
            a, b = dim._args

            return csh.UniformIntegerHyperparameter(
                name=dim.name,
                lower=a,
                upper=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=dim.prior_name == 'int_reciprocal',
            )

        if dim.prior_name in ('norm', 'normal'):
            a, b = dim._args

            return csh.NormalIntegerHyperparameter(
                name=dim.name,
                mu=a,
                sigma=b,
                default_value=dim.default_value,
                q=_qantization(dim),
                log=False,
                lower=dim.low if hasattr(dim, 'low') else None,
                upper=dim.high if hasattr(dim, 'high') else None
            )

        return None

    def categorical(self, dim):
        """Convert a categorical dimension into a configspace equivalent"""
        return csh.CategoricalHyperparameter(
            name=dim.name,
            choices=dim.categories,
            weights=dim._probs,
        )

    def fidelity(self, dim):
        """Ignores fidelity dimension as configspace does not have an equivalent"""
        return None

    def space(self, space):
        """Convert orion space to configspace"""
        cspace = cs.ConfigurationSpace()
        dims = []

        for _, dim in space.items():
            cdim = self.visit(dim)

            if cdim:
                dims.append(cdim)

        cspace.add_hyperparameters(dims)
        return cspace


def toconfigspace(space):
    """Convert orion space to configspace"""
    conversion = ToConfigSpace()
    return conversion.space(space)


def tooriondim(dim):
    """Convert a config space dimension to an orion dimension"""
    klass = Integer
    args = []
    kwargs = dict(
        # default_value=dim.default_value
    )

    if isinstance(dim, (csh.UniformFloatHyperparameter, csh.UniformIntegerHyperparameter)):
        if isinstance(dim, csh.UniformFloatHyperparameter):
            klass = Real

        dist = 'uniform'
        args.append(dim.lower)
        args.append(dim.upper)
        # kwargs['prevision'] = log10(-dim.q)

        if dim.log:
            dist = 'reciprocal'

    if isinstance(dim, (csh.NormalFloatHyperparameter, csh.NormalIntegerHyperparameter)):
        if isinstance(dim, csh.NormalFloatHyperparameter):
            klass = Real

        dist = 'norm'
        args.append(dim.mu)
        args.append(dim.sigma)
        # kwargs['precision'] = log10(-dim.q)

        if dim.lower:
            kwargs['low'] = dim.lower
            kwargs['high'] = dim.upper

    if isinstance(dim, csh.CategoricalHyperparameter):
        klass = Categorical
        choices = {k: w for k, w in zip(dim.choices, dim.probabilities)}
        return klass(dim.name, choices)

    return klass(dim.name, dist, *args, **kwargs)


def toorionspace(cspace):
    """Convert from orion space to configspace"""
    space = Space()

    for _, cdim in cspace.get_hyperparameters_dict().items():
        odim = tooriondim(cdim)
        space.register(odim)

    return space
