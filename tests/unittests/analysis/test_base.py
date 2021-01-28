"""Tests for :mod:`orion.analysis.base`"""
import pytest

from orion.analysis.base import flatten_params
from orion.core.io.space_builder import SpaceBuilder


@pytest.fixture
def space():
    space = SpaceBuilder().build(
        {
            "x": "uniform(0, 6)",
            "y": "uniform(0, 3)",
            "z": "choices(['a', 'b', 'c'])",
        }
    )
    return space


@pytest.fixture
def params():
    return list("xyz")


@pytest.fixture
def hspace():
    space = SpaceBuilder().build(
        {
            "x": "uniform(0, 6, shape=3)",
            "y": "uniform(0, 3, shape=(2, 3))",
            "z": "choices(['a', 'b', 'c'])",
        }
    )
    return space


@pytest.fixture
def flat_params():
    return (
        [f"x[{i}]" for i in range(3)]
        + [f"y[{i},{j}]" for i in range(2) for j in range(3)]
        + ["z"]
    )


class TestFlattenParams:
    def test_no_params(self, space, params):
        """Test that all params are returned if None"""
        assert flatten_params(space) == params

    def test_flat_no_params(self, hspace, flat_params):
        """Test that all flattened params are returned if None"""
        assert flatten_params(hspace) == flat_params

    def test_params_unchanged(self, space):
        """Test that params list passed is not modified"""
        params = ["x", "y"]
        flatten_params(space, params)
        assert params == ["x", "y"]

    def test_unexisting_params(self, space):
        """Test that ValueError is raised if passing unexisting params"""
        with pytest.raises(ValueError) as exc:
            flatten_params(space, ["idoexistbelieveme!!!"])
        assert exc.match(f"Parameter idoexistbelieveme!!! not contained in space: ")

    def test_no_flatten(self, space, hspace):
        """Test selection of params not involving flattening"""
        assert flatten_params(space, ["x", "y"]) == ["x", "y"]
        assert flatten_params(hspace, ["z"]) == ["z"]

    def test_flattened_params(self, hspace):
        """Test selecting specific flattened params"""
        params = ["x[0]", "x[2]", "y[0,2]", "y[1,1]", "z"]
        assert flatten_params(hspace, params) == params

    def test_top_params(self, hspace):
        """Test selecting all flattened keys of a parameter"""
        params = ["x", "y[0,2]", "y[1,1]", "z"]
        assert (
            flatten_params(hspace, params) == [f"x[{i}]" for i in range(3)] + params[1:]
        )
