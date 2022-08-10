import pytest
from pytest import MonkeyPatch

from orion.algo.space import Space
from orion.core.io.space_builder import SpaceBuilder


@pytest.fixture
def no_shutil_copytree(monkeypatch: MonkeyPatch):
    monkeypatch.setattr("shutil.copytree", lambda dir_a, dir_b: None)
    yield


@pytest.fixture
def space() -> Space:
    return SpaceBuilder().build(
        {
            "x": "uniform(0, 100)",
            "y": "uniform(0, 10, discrete=True)",
            "z": 'choices(["a", "b", 0, True])',
            "f": "fidelity(1, 100, base=1)",
        }
    )


@pytest.fixture
def hspace() -> Space:
    return SpaceBuilder().build(
        {
            "numerical": {
                "x": "uniform(0, 100)",
                "y": "uniform(0, 10, discrete=True)",
                "f": "fidelity(1, 100, base=1)",
            },
            "z": 'choices(["a", "b", 0, True])',
        }
    )
