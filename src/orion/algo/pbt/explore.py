import numpy

from orion.core.utils import GenericFactory
from orion.core.utils.flatten import flatten, unflatten


class BaseExplore:
    def __init__(self):
        pass

    def __call__(self, rng, space, params):
        pass

    @property
    def configuration(self):
        return dict(of_type=self.__class__.__name__.lower())


class PipelineExplore(BaseExplore):
    def __init__(self, explore_configs):
        self.pipeline = []
        for explore_config in explore_configs:
            self.pipeline.append(explore_factory.create(**explore_config))

    def __call__(self, rng, space, params):
        for explore in self.pipeline:
            new_params = explore(rng, space, params)
            if new_params is not params:
                return new_params

        return params

    @property
    def configuration(self):
        configuration = super(PipelineExplore, self).configuration
        configuration["explore_configs"] = [
            explore.configuration for explore in self.pipeline
        ]
        return configuration


class PerturbExplore(BaseExplore):
    def __init__(self, factor=1.2, volatility=0.0001):
        self.factor = factor
        self.volatility = volatility

    def perturb_real(self, rng, dim_value, interval):
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
        return dim.sample(1, seed=tuple(rng.randint(0, 1000000, size=3)))[0]

    def __call__(self, rng, space, params):
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
        configuration = super(PerturbExplore, self).configuration
        configuration["factor"] = self.factor
        configuration["volatility"] = self.volatility
        return configuration


class ResampleExplore(BaseExplore):
    def __init__(self, probability=0.2):
        self.probability = probability

    def __call__(self, rng, space, params):
        if rng.random() < self.probability:
            trial = space.sample(1, seed=tuple(rng.randint(0, 1000000, size=3)))[0]
            params = trial.params

        return params

    @property
    def configuration(self):
        configuration = super(ResampleExplore, self).configuration
        configuration["probability"] = self.probability
        return configuration


explore_factory = GenericFactory(BaseExplore)
