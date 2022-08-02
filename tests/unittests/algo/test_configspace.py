import pytest

from orion.algo.space import Categorical, Fidelity, Integer, Real, Space, to_orionspace
from orion.algo.space.configspace import IMPORT_ERROR, UnsupportedPrior, to_configspace

if IMPORT_ERROR:
    pytest.skip("Running without ConfigSpace", allow_module_level=True)


def test_orion_configspace():
    space = Space()

    space.register(Integer("r1i", "reciprocal", 1, 6))
    space.register(Integer("u1i", "uniform", -3, 6))
    space.register(Integer("u2i", "uniform", -3, 6))
    space.register(Integer("u3i", "uniform", -3, 6, default_value=2))

    space.register(Real("r1f", "reciprocal", 1, 6))
    space.register(Real("u1f", "uniform", -3, 6))
    space.register(Real("u2f", "uniform", -3, 6))
    space.register(Real("name.u2f", "uniform", -3, 6))

    space.register(Categorical("c1", ("asdfa", 2)))
    space.register(Categorical("c2", dict(a=0.2, b=0.8)))
    space.register(Fidelity("f1", 1, 9, 3))

    space.register(Real("n1", "norm", 0.9, 0.1, precision=6))
    space.register(Real("n2", "norm", 0.9, 0.1, precision=None))
    space.register(Real("n3", "norm", 0.9, 0.1))
    space.register(Integer("n4", "norm", 1, 2))

    newspace = to_configspace(space)

    roundtrip = to_orionspace(newspace)

    for k, original in space.items():
        # ConfigSpace does not have a fidelity dimension
        # or the alpha prior
        if k in ("f1", "a1i"):
            continue

        converted = roundtrip[k]

        # Orion space did not have default values
        # but ConfigSpace always set them
        if not original.default_value:
            converted._default_value = None

        assert type(original) == type(converted)
        assert original == converted


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
