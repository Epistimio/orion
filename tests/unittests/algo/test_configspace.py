import pytest

from orion.algo.space import Categorical, Fidelity, Integer, Real, Space
from orion.algo.space.configspace import IMPORT_ERROR, to_configspace, configspace_to_orionspace


@pytest.mark.skipif(IMPORT_ERROR, reason="Running without ConfigSpace")
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

    newspace = to_configspace(space)

    roundtrip = configspace_to_orionspace(newspace)

    for k, original in space.items():
        # ConfigSpace does not have a fidelity dimension
        if k == "f1":
            continue

        converted = roundtrip[k]

        # Orion space did not have default values
        # but ConfigSpace always set them
        if not original.default_value:
            converted._default_value = None

        assert type(original) == type(converted)
        assert original == converted
