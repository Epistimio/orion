# -*- coding: utf-8 -*-
"""Tests :func:`orion.analysis.lpi`"""
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

from orion.analysis.lpi import (
    compute_variances,
    lpi,
    make_grid,
    to_numpy,
    train_regressor,
)
from orion.core.io.space_builder import SpaceBuilder

data = pd.DataFrame(
    data={
        "id": ["a", "b", "c", "d"],
        "x": [0, 1, 2, 3],
        "y": [1, 2, 0, 3],
        "objective": [0.1, 0.2, 0.3, 0.5],
    }
)


space = SpaceBuilder().build({"x": "uniform(0, 6)", "y": "uniform(0, 3)"})


def test_accept_empty():
    """Tests an empty dataframe is returned if you give an empty dataframe"""
    empty_frame = pd.DataFrame()
    results = lpi(empty_frame, space)

    assert results.columns.tolist() == ["LPI"]
    assert results.index.tolist() == list(space.keys())
    assert results["LPI"].tolist() == [0, 0]

    empty_frame = pd.DataFrame(columns=["x", "y", "objective"])
    results = lpi(empty_frame, space)

    assert results.columns.tolist() == ["LPI"]
    assert results.index.tolist() == list(space.keys())
    assert results["LPI"].tolist() == [0, 0]


def test_parameter_not_modified():
    """Tests the original dataframe is not modified"""
    original = copy.deepcopy(data)
    lpi(data, space)

    pd.testing.assert_frame_equal(data, original)


def test_to_numpy():
    """Test that trials are correctly converted to numpy array"""
    array = to_numpy(data, space)

    assert array.shape == (4, 3)
    numpy.testing.assert_equal(array[:, 0], data["x"])
    numpy.testing.assert_equal(array[:, 1], data["y"])
    numpy.testing.assert_equal(array[:, 2], data["objective"])


def test_train_regressor():
    """Test training different models"""
    array = to_numpy(data, space)
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


def test_train_regressor_kwargs():
    """Test training models with kwargs"""
    array = to_numpy(data, space)
    model = train_regressor(
        "RandomForestRegressor", array, max_depth=2, max_features="sqrt"
    )
    assert model.max_depth == 2
    assert model.max_features == "sqrt"


def test_train_regressor_invalid():
    """Test error message for invalid model names"""
    array = to_numpy(data, space)
    with pytest.raises(ValueError) as exc:
        train_regressor("IDontExist", array)
    assert exc.match("IDontExist is not a supported regressor")


def test_make_grid():
    """Test grid has correct format"""
    trials = to_numpy(data, space)
    model = train_regressor("RandomForestRegressor", trials)
    best_point = trials[numpy.argmin(trials[:, -1])]
    grid = make_grid(best_point, space, model, 4)

    # Are fixed to anchor value
    numpy.testing.assert_equal(grid[0][:, 1], best_point[1])
    numpy.testing.assert_equal(grid[1][:, 0], best_point[0])

    # Is a grid in search space
    numpy.testing.assert_equal(grid[0][:, 0], [0, 2, 4, 6])
    numpy.testing.assert_equal(grid[1][:, 1], [0, 1, 2, 3])


def test_make_grid_predictor(monkeypatch):
    """Test grid contains corresponding predictions from the model"""
    trials = to_numpy(data, space)
    model = train_regressor("RandomForestRegressor", trials)
    best_point = trials[numpy.argmin(trials[:, -1])]

    # Make sure model is not predicting exactly the original objective
    with numpy.testing.assert_raises(AssertionError):
        numpy.testing.assert_equal(
            best_point[-1], model.predict(best_point[:-1].reshape(1, -1))
        )

    grid = make_grid(best_point, space, model, 4)

    # Verify that grid predictions are those of the model
    numpy.testing.assert_equal(grid[0][:, -1], model.predict(grid[0][:, :-1]))
    numpy.testing.assert_equal(grid[1][:, -1], model.predict(grid[1][:, :-1]))

    # Verify model predictions differ on different points
    with numpy.testing.assert_raises(AssertionError):
        numpy.testing.assert_equal(grid[0][:, -1], grid[1][:, -1])


def test_compute_variance():
    """Test variance computation over the grid"""
    grid = numpy.arange(3 * 5 * 4).reshape(3, 5, 4)

    grid[0, :, -1] = 10
    grid[1, :, -1] = [0, 1, 2, 3, 4]
    grid[2, :, -1] = [0, 10, 20, 30, 40]

    variances = compute_variances(grid)
    assert variances.shape == (3,)
    assert variances[0] == 0
    assert variances[1] == numpy.var([0, 1, 2, 3, 4])
    assert variances[2] == numpy.var([0, 10, 20, 30, 40])


def test_lpi_results():
    """Verify LPI results in DataFrame"""
    results = lpi(data, space, random_state=1)
    assert results.columns.tolist() == ["LPI"]
    assert results.index.tolist() == list(space.keys())
    # The data is made such that x correlates more strongly with objective than y
    assert results["LPI"].loc["x"] > results["LPI"].loc["y"]


def test_lpi_with_categorical_data():
    """Verify LPI can be computed on categorical dimensions"""
    data = pd.DataFrame(
        data={
            "id": ["a", "b", "c", "d"],
            "x": [0, 1, 2, 3],
            "y": ["b", "c", "a", "d"],
            "objective": [0.1, 0.2, 0.3, 0.5],
        }
    )

    space = SpaceBuilder().build(
        {"x": "uniform(0, 6)", "y": 'choices(["a", "b", "c", "d"])'}
    )

    results = lpi(data, space, random_state=1)
    assert results.columns.tolist() == ["LPI"]
    assert results.index.tolist() == ["x", "y"]
    # The data is made such that x correlates more strongly with objective than y
    assert results["LPI"].loc["x"] > results["LPI"].loc["y"]


def test_lpi_with_multidim_data():
    """Verify LPI can be computed on categorical dimensions"""
    data = pd.DataFrame(
        data={
            "id": ["a", "b", "c", "d"],
            "x": [[0, 2, 4], [1, 1, 3], [2, 2, 2], [3, 0, 3]],
            "y": [["b", "b"], ["c", "b"], ["a", "a"], ["d", "c"]],
            "objective": [0.1, 0.2, 0.3, 0.5],
        }
    )

    space = SpaceBuilder().build(
        {"x": "uniform(0, 6, shape=3)", "y": 'choices(["a", "b", "c", "d"], shape=2)'}
    )

    results = lpi(data, space, random_state=1)
    assert results.columns.tolist() == ["LPI"]
    assert results.index.tolist() == ["x[0]", "x[1]", "x[2]", "y[0]", "y[1]"]
    # The data is made such some x correlates more strongly with objective than other x and most y
    assert results["LPI"].loc["x[0]"] > results["LPI"].loc["x[1]"]
    assert results["LPI"].loc["x[1]"] > results["LPI"].loc["x[2]"]
    assert results["LPI"].loc["x[0]"] > results["LPI"].loc["y[0]"]
    assert results["LPI"].loc["x[0]"] > results["LPI"].loc["y[1]"]
