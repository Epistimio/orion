"""Tests for :mod:`orion.analysis.base`"""
import numpy
import pandas as pd
import pytest
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from orion.analysis.base import flatten_numpy, flatten_params, to_numpy, train_regressor
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.transformer import build_required_space

data = pd.DataFrame(
    data={
        "id": ["a", "b", "c"],
        "x": [0, 1, 2],
        "y": [1, 2, 0],
        "z": ["a", "b", "c"],
        "objective": [0.1, 0.2, 0.3],
    }
)


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
def fspace(space):
    return build_required_space(
        space,
        dist_requirement="linear",
        type_requirement="numerical",
        shape_requirement="flattened",
    )


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


def test_to_numpy(space):
    """Test that trials are correctly converted to numpy array"""
    array = to_numpy(data, space)

    assert array.shape == (3, 4)
    numpy.testing.assert_equal(array[:, 0], data["x"])
    numpy.testing.assert_equal(array[:, 1], data["y"])
    numpy.testing.assert_equal(array[:, 2], data["z"])
    numpy.testing.assert_equal(array[:, 3], data["objective"])


class TestTrainRegressor:
    def test_train_regressor(self, space, fspace):
        """Test training different models"""
        array = flatten_numpy(to_numpy(data, space), fspace)
        model = train_regressor("AdaBoostRegressor", array)
        assert isinstance(model, AdaBoostRegressor)
        model = train_regressor("BaggingRegressor", array)
        assert isinstance(model, BaggingRegressor)
        model = train_regressor("ExtraTreesRegressor", array)
        assert isinstance(model, ExtraTreesRegressor)
        model = train_regressor("GradientBoostingRegressor", array)
        assert isinstance(model, GradientBoostingRegressor)
        model = train_regressor("RandomForestRegressor", array)
        assert isinstance(model, RandomForestRegressor)

    def test_train_regressor_kwargs(self, space, fspace):
        """Test training models with kwargs"""
        array = flatten_numpy(to_numpy(data, space), fspace)
        model = train_regressor(
            "RandomForestRegressor", array, max_depth=2, max_features="sqrt"
        )
        assert model.max_depth == 2
        assert model.max_features == "sqrt"

    def test_train_regressor_invalid(self, space, fspace):
        """Test error message for invalid model names"""
        array = flatten_numpy(to_numpy(data, space), fspace)
        with pytest.raises(ValueError) as exc:
            train_regressor("IDontExist", array)
        assert exc.match("IDontExist is not a supported regressor")
