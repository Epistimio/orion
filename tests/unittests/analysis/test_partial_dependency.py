# -*- coding: utf-8 -*-
"""Tests :func:`orion.analysis.partial_dependency`"""
import copy

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

from orion.analysis.base import flatten_numpy, to_numpy, train_regressor
from orion.analysis.partial_dependency_utils import (
    make_grid,
    partial_dependency,
    partial_dependency_grid,
    reverse,
)
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.transformer import build_required_space

data = pd.DataFrame(
    data={
        "id": ["a", "b", "c", "d"],
        "x": [0, 1, 2, 3],
        "y": list(numpy.e ** numpy.array([2, 3, 1, 4])),
        "z": ["a", "a", "b", "c"],
        "objective": [0.1, 0.2, 0.3, 0.5],
    }
)


hdata = pd.DataFrame(
    data={
        "id": ["a", "b", "c", "d"],
        "x": [0, 1, 2, 3],
        "y": list(numpy.e ** numpy.array([[2, 1, 2], [3, 0, 1], [1, 3, 0], [4, 4, 0]])),
        "z": ["a", "a", "b", "c"],
        "objective": [0.1, 0.2, 0.3, 0.5],
    }
)


@pytest.fixture
def space():
    """Base flat space"""
    return SpaceBuilder().build(
        {
            "x": "uniform(0, 6)",
            "y": "loguniform(0.001, 1)",
            "z": "choices(['a', 'b', 'c'])",
        }
    )


@pytest.fixture
def hspace():
    """Space with multidim dims"""
    return SpaceBuilder().build(
        {
            "x": "uniform(0, 6)",
            "y": f"loguniform({numpy.e}, {numpy.e}**4, shape=3)",
            "z": "choices(['a', 'b', 'c'])",
        }
    )


def flatten_space(some_space):
    """Flatten a space"""
    return build_required_space(
        some_space,
        dist_requirement="linear",
        type_requirement="numerical",
        shape_requirement="flattened",
    )


def mock_model():
    """Return a mocked regressor which just predict iterated integers"""

    class Model:
        """Mocked Regressor"""

        def __init__(self):
            self.i = 0

        def predict(self, data):
            """Returns counting of predictions requested."""
            data = numpy.arange(data.shape[0]) + self.i
            self.i += data.shape[0]
            return data  #  + numpy.random.normal(0, self.i, size=data.shape[0])

    return Model()


def mock_train_regressor(monkeypatch, assert_model=None):
    """Mock the train_regressor to return the mocked regressor instead"""

    def train_regressor(model, data, **kwargs):
        """Return the mocked model, and then model argument if requested"""
        if assert_model:
            assert model == assert_model
        return mock_model()

    monkeypatch.setattr(
        "orion.analysis.partial_dependency_utils.train_regressor", train_regressor
    )


def test_reverse_no_reshape_needed(space):
    """Test that transformed grids can be reversed to original type, when there was no reshape"""
    flattened_space = flatten_space(space)

    grid = {
        "x": [0, 1, 2, 4],
        "y": numpy.log([0.001, 0.01, 0.1, 1]),
        "z": [0, 0, 1, 2],
    }

    reversed_grid = reverse(flattened_space, grid)
    assert reversed_grid["x"] == [0, 1, 2, 4]
    assert reversed_grid["y"] == [0.001, 0.01, 0.1, 1]
    assert reversed_grid["z"] == ["a", "a", "b", "c"]


def test_reverse_with_reshape(hspace):
    """Test that transformed grids can be reversed to original type, when reshape was applied"""
    flattened_space = flatten_space(hspace)

    grid = {
        "x": [0, 1, 2, 4],
        "y[0]": numpy.log([0.001, 0.01, 0.1, 1]),
        "y[1]": numpy.log([1, 0.01, 0.1, 0.001]),
        "z": [0, 0, 1, 2],
    }

    reversed_grid = reverse(flattened_space, grid)
    assert reversed_grid["x"] == [0, 1, 2, 4]
    assert reversed_grid["y[0]"] == [0.001, 0.01, 0.1, 1]
    assert reversed_grid["y[1]"] == [1, 0.01, 0.1, 0.001]
    assert reversed_grid["z"] == ["a", "a", "b", "c"]


def test_make_grid(hspace):
    """Test that different dimension types are supported"""
    flattened_space = flatten_space(hspace)
    assert list(make_grid(flattened_space["x"], 7)) == [0, 1, 2, 3, 4, 5, 6]
    numpy.testing.assert_almost_equal(
        make_grid(flattened_space["y[0]"], 4), [1, 2, 3, 4], decimal=4
    )
    numpy.testing.assert_almost_equal(
        make_grid(flattened_space["y[2]"], 4), [1, 2, 3, 4], decimal=4
    )
    assert list(make_grid(flattened_space["z"], 10)) == [0, 1, 2]


def test_partial_dependency_grid(hspace):
    """Test the computation of the averages and stds"""

    flattened_space = flatten_space(hspace)

    n_points = 5
    n_samples = 20
    samples = flattened_space.sample(n_samples)
    samples = pd.DataFrame(samples, columns=flattened_space.keys())

    params = ["x", "y[0]", "y[2]", "z"]

    # Test for 1 param
    grid, averages, stds = partial_dependency_grid(
        flattened_space, mock_model(), ["x"], samples, n_points=n_points
    )

    assert list(grid.keys()) == ["x"]
    assert list(grid["x"]) == [0, 1.5, 3, 4.5, 6]
    assert averages.shape == (n_points,)
    assert stds.shape == (n_points,)
    assert averages[0] == numpy.arange(n_samples).mean()
    assert (
        averages[4]
        == numpy.arange(n_samples * (n_points - 1), n_samples * n_points).mean()
    )
    assert stds[0] == numpy.arange(n_samples).std()

    # Test for 2 param
    grid, averages, stds = partial_dependency_grid(
        flattened_space, mock_model(), ["x", "y[0]"], samples, n_points=n_points
    )

    assert list(grid.keys()) == ["x", "y[0]"]
    assert list(grid["x"]) == [0, 1.5, 3, 4.5, 6]
    # assert list(grid["y[0]"]) == [0, 0.75, 1.5, 2.25, 3]
    numpy.testing.assert_almost_equal(
        grid["y[0]"],
        numpy.linspace(
            numpy.log(float(f"{numpy.e}")),
            numpy.log(float(f"{numpy.e}") ** 4),
            num=n_points,
        ),
        decimal=4,
    )

    assert averages.shape == (n_points, n_points)
    assert stds.shape == (n_points, n_points)
    assert averages[0, 0] == numpy.arange(n_samples).mean()
    assert (
        averages[4, 4]
        == numpy.arange(
            n_samples * n_points * n_points - n_samples, n_samples * n_points * n_points
        ).mean()
    )
    assert stds[0, 0] == numpy.arange(n_samples).std()
    assert stds[4, 4] == numpy.arange(n_samples).std()

    # Test for 2 param with one categorical, with less categories then n_points
    grid, averages, stds = partial_dependency_grid(
        flattened_space, mock_model(), ["x", "z"], samples, n_points=n_points
    )

    assert list(grid.keys()) == ["x", "z"]
    assert list(grid["x"]) == [0, 1.5, 3, 4.5, 6]
    assert list(grid["z"]) == [0, 1, 2]
    assert averages.shape == (3, n_points)
    assert stds.shape == (3, n_points)

    assert averages[0, 0] == numpy.arange(n_samples).mean()
    assert (
        averages[2, 4]
        == numpy.arange(
            n_samples * 3 * n_points - n_samples, n_samples * 3 * n_points
        ).mean()
    )
    assert stds[0, 0] == numpy.arange(n_samples).std()
    assert stds[2, 4] == numpy.arange(n_samples).std()


def test_accept_empty(space):
    """Tests an empty dataframe is returned if you give an empty dataframe"""
    empty_frame = pd.DataFrame()
    results = partial_dependency(empty_frame, space, n_grid_points=3, n_samples=5)

    assert results == {}

    empty_frame = pd.DataFrame(columns=["x", "y", "z", "objective"])
    results = partial_dependency(empty_frame, space, n_grid_points=3, n_samples=5)

    assert results == {}


def test_parameter_not_modified(monkeypatch, space):
    """Tests the original dataframe is not modified"""
    original = copy.deepcopy(data)

    mock_train_regressor(monkeypatch)
    partial_dependency(data, space, n_samples=10)

    pd.testing.assert_frame_equal(data, original)


def test_single_param(monkeypatch, space):
    """Test computing for a single param"""
    mock_train_regressor(monkeypatch)

    partial_dependencies = partial_dependency(data, space, params=["x"], n_samples=10)
    assert list(partial_dependencies.keys()) == ["x"]
    assert len(partial_dependencies["x"][0]["x"]) == 10
    assert partial_dependencies["x"][1].shape == (10,)
    assert partial_dependencies["x"][2].shape == (10,)


def test_multiple_params(monkeypatch, space):
    """Test computing for multiple param"""
    mock_train_regressor(monkeypatch)

    partial_dependencies = partial_dependency(
        data, space, params=["x", "y"], n_samples=10
    )
    assert set(partial_dependencies.keys()) == {"x", ("x", "y"), "y"}
    assert len(partial_dependencies["x"][0]["x"]) == 10
    assert len(partial_dependencies["y"][0]["y"]) == 10
    assert len(partial_dependencies[("x", "y")][0]["x"]) == 10
    assert len(partial_dependencies[("x", "y")][0]["y"]) == 10
    assert partial_dependencies["x"][1].shape == (10,)
    assert partial_dependencies["x"][2].shape == (10,)
    assert partial_dependencies["y"][1].shape == (10,)
    assert partial_dependencies["y"][2].shape == (10,)
    assert partial_dependencies[("x", "y")][1].shape == (10, 10)
    assert partial_dependencies[("x", "y")][2].shape == (10, 10)


def test_multidim(monkeypatch, hspace):
    """Test computing for multiple params in multidim space"""
    mock_train_regressor(monkeypatch)

    partial_dependencies = partial_dependency(
        hdata, hspace, params=["x", "y[0]", "y[1]"], n_samples=10
    )
    assert set(partial_dependencies.keys()) == {
        "x",
        "y[0]",
        "y[1]",
        ("x", "y[0]"),
        ("x", "y[1]"),
        ("y[0]", "y[1]"),
    }
    assert len(partial_dependencies["x"][0]["x"]) == 10
    assert len(partial_dependencies["y[0]"][0]["y[0]"]) == 10
    assert len(partial_dependencies["y[1]"][0]["y[1]"]) == 10
    assert len(partial_dependencies[("x", "y[0]")][0]["x"]) == 10
    assert len(partial_dependencies[("x", "y[0]")][0]["y[0]"]) == 10
    assert partial_dependencies["x"][1].shape == (10,)
    assert partial_dependencies["x"][2].shape == (10,)
    assert partial_dependencies["y[0]"][1].shape == (10,)
    assert partial_dependencies["y[0]"][2].shape == (10,)
    assert partial_dependencies[("x", "y[0]")][1].shape == (10, 10)
    assert partial_dependencies[("x", "y[0]")][2].shape == (10, 10)


def test_categorical(monkeypatch, space):
    """Test computing for categorical"""
    mock_train_regressor(monkeypatch)

    partial_dependencies = partial_dependency(data, space, params=["z"], n_samples=10)

    assert set(partial_dependencies.keys()) == {"z"}
    assert partial_dependencies["z"][0]["z"] == ["a", "b", "c"]
    assert partial_dependencies["z"][1].shape == (3,)
    assert partial_dependencies["z"][2].shape == (3,)


def test_n_grid_points(monkeypatch, space):
    """Test that number of points on grid is correct, including adjustment for categorical dims"""
    mock_train_regressor(monkeypatch)

    n_grid_points = 5
    partial_dependencies = partial_dependency(
        data, space, params=["x", "z"], n_grid_points=n_grid_points, n_samples=10
    )

    assert set(partial_dependencies.keys()) == {"x", ("x", "z"), "z"}
    assert partial_dependencies["z"][1].shape == (3,)
    assert partial_dependencies["z"][2].shape == (3,)

    assert len(partial_dependencies["x"][0]["x"]) == n_grid_points
    assert len(partial_dependencies[("x", "z")][0]["x"]) == n_grid_points
    assert partial_dependencies[("x", "z")][0]["z"] == ["a", "b", "c"]
    assert partial_dependencies["z"][0]["z"] == ["a", "b", "c"]
    assert partial_dependencies["x"][1].shape == (n_grid_points,)
    assert partial_dependencies["x"][2].shape == (n_grid_points,)
    assert partial_dependencies["z"][1].shape == (3,)
    assert partial_dependencies["z"][2].shape == (3,)
    assert partial_dependencies[("x", "z")][1].shape == (3, n_grid_points)
    assert partial_dependencies[("x", "z")][2].shape == (3, n_grid_points)


def test_n_samples(monkeypatch, space):
    """Test n_samples is used properly when sampling random points"""
    mock_train_regressor(monkeypatch)

    PARAMS = ["x", "z"]
    N_SAMPLES = numpy.random.randint(20, 50)

    def mock_partial_dependency_grid(space, model, params, samples, n_points):
        assert samples.shape == (N_SAMPLES, len(space))
        return partial_dependency_grid(space, model, params, samples, n_points)

    monkeypatch.setattr(
        "orion.analysis.partial_dependency_utils.partial_dependency_grid",
        mock_partial_dependency_grid,
    )

    partial_dependency(data, space, params=PARAMS, n_grid_points=5, n_samples=N_SAMPLES)
