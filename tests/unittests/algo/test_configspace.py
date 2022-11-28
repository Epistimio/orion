import pytest

from orion.algo.space import Categorical, Fidelity, Integer, Real, Space, to_orionspace
from orion.algo.space.configspace import IMPORT_ERROR, UnsupportedPrior, to_configspace

if IMPORT_ERROR:
    pytest.skip("Running without ConfigSpace", allow_module_level=True)


def compare_spaces(s1, s2):
    for k, original in s1.items():
        # ConfigSpace does not have a fidelity dimension
        # or the alpha prior
        if k in ("f1", "a1i"):
            continue

        converted = s2[k]

        # Orion space did not have default values
        # but ConfigSpace always set them
        if not original.default_value:
            converted._default_value = None

        assert type(original) == type(converted)
        assert original == converted


def test_orion_configspace():
    space = Space()

    # NB: scipy uniform [loc, scale], configspace [min, max] with max = loc + scale, loc = min
    def uniform(type, name, low, high, **kwargs):
        return type(name, "uniform", low, high - low, **kwargs)

    space.register(Integer("r1i", "reciprocal", 1, 6))
    space.register(uniform(Integer, "u1i", -3, 6))
    space.register(uniform(Integer, "u2i", -3, 6))
    space.register(uniform(Integer, "u4i", -4, 0, default_value=-1))
    space.register(uniform(Integer, "u3i", -3, 6, default_value=2))

    space.register(Real("r1f", "reciprocal", 1, 6))
    space.register(uniform(Real, "u1f", -3, 6))
    space.register(uniform(Real, "u2f", -3, 6))
    space.register(uniform(Real, "u4f", -4, 0, default_value=-0.2))
    space.register(uniform(Real, "name.u2f", -3, 6))

    space.register(Categorical("c1", ("asdfa", 2)))
    space.register(Categorical("c2", dict(a=0.2, b=0.8)))
    space.register(Fidelity("f1", 1, 9, 3))

    space.register(Real("n1", "norm", 0.9, 0.1, precision=6))
    space.register(Real("n2", "norm", 0.9, 0.1, precision=None))
    space.register(Real("n3", "norm", 0.9, 0.1))
    space.register(Integer("n4", "norm", 1, 2))

    newspace = to_configspace(space)
    roundtrip = to_orionspace(newspace)

    compare_spaces(space, roundtrip)


def test_orion_configspace_kwargs():
    space = Space()

    # NB: scipy uniform [loc, scale], configspace [min, max] with max = loc + scale, loc = min
    def uniform(type, name, low, high, **kwargs):
        return type(name, "uniform", loc=low, scale=high - low, **kwargs)

    space.register(Integer("r2i", "reciprocal", a=1, b=6))
    space.register(uniform(Integer, "u1i", -3, 6))
    space.register(uniform(Integer, "u2i", -3, 6))
    space.register(uniform(Integer, "u4i", -4, 0, default_value=-1))
    space.register(uniform(Integer, "u3i", -3, 6, default_value=2))

    space.register(Real("r1f", "reciprocal", a=1, b=6))
    space.register(uniform(Real, "u1f", -3, 6))
    space.register(uniform(Real, "u2f", -3, 6))
    space.register(uniform(Real, "u4f", -4, 0, default_value=-0.2))
    space.register(uniform(Real, "name.u2f", -3, 6))

    space.register(Categorical("c1", categories=("asdfa", 2)))
    space.register(Categorical("c2", categories=dict(a=0.2, b=0.8)))
    space.register(Fidelity("f1", low=1, high=9, base=3))

    space.register(Real("n1", "norm", loc=0.9, scale=0.1, precision=6))
    space.register(Real("n2", "norm", loc=0.9, scale=0.1, precision=None))
    space.register(Real("n3", "norm", loc=0.9, scale=0.1))
    space.register(Integer("n4", "norm", loc=1, scale=2))

    newspace = to_configspace(space)
    r1 = to_orionspace(newspace)

    newspace = to_configspace(r1)
    r2 = to_orionspace(newspace)

    # the first roundtrip conversion converted kwargs to positional arguments
    # second roundtrip should be exactly the same
    compare_spaces(r1, r2)

    for k, original in space.items():
        dim1 = r1.get(k)
        dim2 = r2.get(k)

        print(f"- {k:>10}", original)
        print(" " * 12, dim1)
        print(" " * 12, dim2)
        print()


def test_configspace_to_orion_unsupported():
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.hyperparameters import OrdinalHyperparameter

    cspace = ConfigurationSpace()
    cspace.add_hyperparameters([OrdinalHyperparameter("a", (1, 2, 0, 3))])

    with pytest.raises(NotImplementedError):
        _ = to_orionspace(cspace)


def test_orion_configspace_unsupported():
    space = Space()
    space.register(Integer("a1i", "alpha", 1, 6))

    with pytest.raises(UnsupportedPrior):
        _ = to_configspace(space)
