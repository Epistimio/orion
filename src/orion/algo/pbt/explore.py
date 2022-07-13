"""
Explore classes for Population Based Training
---------------------------------------------

Formulation of a general explore function for population based training.
Implementations must inherit from ``orion.algo.pbt.BaseExplore``.

Explore objects can be created using `explore_factory.create()`.

Examples
--------
>>> explore_factory.create('PerturbExplore')
>>> explore_factory.create('PerturbExplore', factor=1.5)

"""

import numpy

from orion.core.utils import GenericFactory
from orion.core.utils.flatten import flatten, unflatten


class BaseExplore:
    """Abstract class for Explore in :py:class:`orion.algo.pbt.pbt.PBT`

    The explore class is responsible for proposing new parameters for a given trial and space.

    This class is expected to be stateless and serve as a configurable callable object.
    """

    def __init__(self):
        pass

    def __call__(self, rng, space, params):
        """Execute explore

        The method receives the space and the parameters of the current trial under examination.
        It must then select new parameters for the trial.

        Parameters
        ----------
        rng: numpy.random.Generator
            A random number generator. It is not contained in ``BaseExplore`` because the explore
            class must be stateless.
        space: Space
            The search space optimized by the algorithm.
        params: dict
            Dictionary representing the parameters of the current trial under examination
            (`trial.params`).

        Returns
        -------
        ``dict``
            The new set of parameters for the trial to be branched.

        """

    @property
    def configuration(self):
        """Configuration of the exploit object"""
        return dict(of_type=self.__class__.__name__.lower())


class PipelineExplore(BaseExplore):
    """
    Pipeline of BaseExploit objects

    The pipeline executes the BaseExplore objects sequentially. If one object returns
    the parameters that are different than the ones passed (``params``), then the pipeline
    returns these parameter values. Otherwise, if all BaseExplore objects return the same
    parameters as the one passed to the pipeline, then the pipeline returns it.

    Parameters
    ----------
    explore_configs: list of dict
        List of dictionary representing the configurations of BaseExplore children.

    Examples
    --------
    This pipeline is useful if for instance you want to sample from the space with a small
    probability, but otherwise use a local perturbation.

    >>> PipelineExplore(
        explore_configs=[
            {'of_type': 'ResampleExplore', probability=0.05},
            {'of_type': 'PerturbExplore'}
        ])

    """

    def __init__(self, explore_configs):
        self.pipeline = []
        for explore_config in explore_configs:
            self.pipeline.append(explore_factory.create(**explore_config))

    def __call__(self, rng, space, params):
        """Execute explore objects sequentially

        If one explore object returns the parameters that are different than the ones passed
        (``params``), then the pipeline returns these parameter values. Otherwise, if all
        BaseExplore objects return the same parameters as the one passed to the pipeline, then the
        pipeline returns it.

        Parameters
        ----------
        rng: numpy.random.Generator
            A random number generator. It is not contained in ``BaseExplore`` because the explore
            class must be stateless.
        space: Space
            The search space optimized by the algorithm.
        params: dict
            Dictionary representing the parameters of the current trial under examination
            (`trial.params`).

        Returns
        -------
        ``dict``
            The new set of parameters for the trial to be branched.

        """

        for explore in self.pipeline:
            new_params = explore(rng, space, params)
            if new_params is not params:
                return new_params

        return params

    @property
    def configuration(self):
        """Configuration of the exploit object"""
        configuration = super().configuration
        configuration["explore_configs"] = [
            explore.configuration for explore in self.pipeline
        ]
        return configuration


class PerturbExplore(BaseExplore):
    """
    Perturb parameters for exploration

    Given a set of parameter values, this exploration object randomly perturb
    them with a given ``factor``. It will multiply the value of a dimension
    with probability 0.5, otherwise divide it. Values are clamped to limits of the
    search space when exceeding it. For categorical dimensions, a new value is sampled
    from categories with equal probability for each categories.

    Parameters
    ----------
    factor: float, optional
        Factor used to multiply or divide with probability 0.5 the values of the dimensions.
        Only applies to real or int dimensions. Integer dimensions are pushed to next integer
        if ``new_value > value`` otherwise reduced to previous integer, where new_value is
        the result of either ``value * factor`` or ``value / factor``.
        Categorial dimensions are sampled from categories randomly. Default: 1.2
    volatility: float, optional
        If the results of ``value * factor`` or ``value / factor`` exceeds the
        limit of the search space, the new value is set to limit and then added
        or subtracted ``abs(normal(0, volatility))`` (if at lower limit or upper limit).
        Default: 0.0001

    Notes
    -----
    Categorical dimensions with special probabilities are not supported for now. A category
    with be sampled with equal probability for each categories.

    """

    def __init__(self, factor=1.2, volatility=0.0001):
        self.factor = factor
        self.volatility = volatility

    def perturb_real(self, rng, dim_value, interval):
        """Perturb real value dimension

        Parameters
        ----------
        rng: numpy.random.Generator
            Random number generator
        dim_value: float
            Value of the dimension
        interval: tuple of float
            Limit of the dimension (lower, upper)

        """
        if rng.random() > 0.5:
            dim_value *= self.factor
        else:
            dim_value *= 1.0 / self.factor

        if dim_value > interval[1]:
            dim_value = max(
                interval[1] - numpy.abs(rng.normal(0, self.volatility)), interval[0]
            )
        elif dim_value < interval[0]:
            dim_value = min(
                interval[0] + numpy.abs(rng.normal(0, self.volatility)), interval[1]
            )

        return dim_value

    def perturb_int(self, rng, dim_value, interval):
        """Perturb integer value dimension

        Parameters
        ----------
        rng: numpy.random.Generator
            Random number generator
        dim_value: int
            Value of the dimension
        interval: tuple of int
            Limit of the dimension (lower, upper)

        """

        new_dim_value = self.perturb_real(rng, dim_value, interval)

        rounded_new_dim_value = int(numpy.round(new_dim_value))

        if rounded_new_dim_value == dim_value and new_dim_value > dim_value:
            new_dim_value = dim_value + 1
        elif rounded_new_dim_value == dim_value and new_dim_value < dim_value:
            new_dim_value = dim_value - 1
        else:
            new_dim_value = rounded_new_dim_value

        # Avoid out of dimension.
        new_dim_value = min(max(new_dim_value, interval[0]), interval[1])

        return new_dim_value

    def perturb_cat(self, rng, dim_value, dim):
        """Perturb categorical dimension

        Parameters
        ----------
        rng: numpy.random.Generator
            Random number generator
        dim_value: object
            Value of the dimension, can be any type.
        dim: orion.algo.space.CategoricalDimension
            CategoricalDimension object defining the search space for this dimension.

        """
        # NOTE: Can't use rng.choice otherwise the interval is an array of strings
        #       and the value is casted.
        choices = dim.interval()
        choice = choices[rng.randint(len(choices))]
        return choice

    def __call__(self, rng, space, params):
        """Execute perturbation

        Given a set of parameter values, this exploration object randomly perturb them with a given
        ``factor``. It will multiply the value of a dimension with probability 0.5, otherwise divide
        it. Values are clamped to limits of the search space when exceeding it. For categorical
        dimensions, a new value is sampled from categories with equal probability for each
        categories.

        Parameters
        ----------
        rng: numpy.random.Generator
            A random number generator. It is not contained in ``BaseExplore`` because the explore
            class must be stateless.
        space: Space
            The search space optimized by the algorithm.
        params: dict
            Dictionary representing the parameters of the current trial under examination
            (`trial.params`).

        Returns
        -------
        ``dict``
            The new set of parameters for the trial to be branched.

        """

        new_params = {}
        params = flatten(params)
        for dim in space.values():
            dim_value = params[dim.name]
            if dim.type == "real":
                dim_value = self.perturb_real(rng, dim_value, dim.interval())
            elif dim.type == "integer":
                dim_value = self.perturb_int(rng, dim_value, dim.interval())
            elif dim.type == "categorical":
                dim_value = self.perturb_cat(rng, dim_value, dim)
            elif dim.type == "fidelity":
                # do nothing
                pass
            else:
                raise ValueError(f"Unsupported dimension type {dim.type}")

            new_params[dim.name] = dim_value

        return unflatten(new_params)

    @property
    def configuration(self):
        """Configuration of the exploit object"""
        configuration = super().configuration
        configuration["factor"] = self.factor
        configuration["volatility"] = self.volatility
        return configuration


class ResampleExplore(BaseExplore):
    """
    Sample parameters search space

    With given probability ``probability``, it will sample a new set of parameters
    from the search space totally independently of the ``parameters`` passed to ``__call__``.
    Otherwise, it will return the passed ``parameters``.

    Parameters
    ----------
    probability: float, optional
        Probability of sampling a new set of parameters. Default: 0.2

    """

    def __init__(self, probability=0.2):
        self.probability = probability

    def __call__(self, rng, space, params):
        """Execute resampling

        With given probability ``self.probability``, it will sample a new set of parameters from the
        search space totally independently of the ``parameters`` passed to ``__call__``.  Otherwise,
        it will return the passed ``parameters``.

        Parameters
        ----------
        rng: numpy.random.Generator
            A random number generator. It is not contained in ``BaseExplore`` because the explore
            class must be stateless.
        space: Space
            The search space optimized by the algorithm.
        params: dict
            Dictionary representing the parameters of the current trial under examination
            (`trial.params`).

        Returns
        -------
        ``dict``
            The new set of parameters for the trial to be branched.

        """

        if rng.random() < self.probability:
            trial = space.sample(1, seed=tuple(rng.randint(0, 1000000, size=3)))[0]
            params = trial.params

        return params

    @property
    def configuration(self):
        """Configuration of the exploit object"""
        configuration = super().configuration
        configuration["probability"] = self.probability
        return configuration


explore_factory = GenericFactory(BaseExplore)
