# -*- coding: utf-8 -*-
"""Tests :func:`orion.analysis.regret`"""
import pandas as pd
import pytest

from orion.analysis import regret


def test_accept_dict():
    """Tests dictionary parameter is supported"""
    data = {"id": ["a", "b", "c", "d"], "objective": [0.1, 0.2, 0.3, 0.5]}

    result = regret(data)

    assert type(result) is pd.DataFrame


def test_accept_empty():
    """Tests an empty dataframe is returned if you give an empty dataframe"""
    empty_frame = pd.DataFrame()
    result = regret(empty_frame)

    assert result.empty
    assert result.equals(empty_frame)

    empty_frame = pd.DataFrame(columns=["id", "objective"])
    result = regret(empty_frame)

    assert result.empty
    assert "best" not in result.columns
    assert "best_id" not in result.columns
    assert result.equals(empty_frame)


def test_parameter_not_modified():
    """Tests the original dataframe is not modified"""
    data = pd.DataFrame(
        data={"id": ["a", "b", "c", "d"], "objective": [0.1, 0.2, 0.3, 0.5]}
    )

    result = regret(data)

    assert len(data.columns) == 2
    assert len(result.columns) == 4


def test_length_names():
    """Tests the length of `names` parameter is two"""
    data = pd.DataFrame(data={"id": ["a", "b"], "objective": [0.1, 0.2]})

    with pytest.raises(ValueError) as exception:
        regret(data, names=())
    assert "`names` requires a tuple with 2 elements. 0 provided." in str(
        exception.value
    )

    with pytest.raises(ValueError) as exception:
        regret(data, names=("a", "b", "c"))
    assert "`names` requires a tuple with 2 elements. 3 provided." in str(
        exception.value
    )


def test_default_column_names():
    """Tests default column names"""
    data = pd.DataFrame(
        data={"id": ["a", "b", "c", "d"], "objective": [0.1, 0.2, 0.3, 0.5]}
    )

    result = regret(data)

    assert "best" in result.columns
    assert "best_id" in result.columns


def test_input_column_names():
    """Tests custom column names"""
    data = pd.DataFrame(
        data={"id": ["a", "b", "c", "d"], "objective": [0.1, 0.2, 0.3, 0.5]}
    )

    result = regret(data, names=("something", "else"))

    assert "something" in result.columns
    assert "else" in result.columns


def test_regret_sequential():
    """Tests that cumulative best objectives are linked to their respective ids"""
    data = pd.DataFrame(
        data={
            "id": ["a", "b", "c", "d"],
            "objective": [10, 12, 8, 9],
        }
    )

    expected_best = [10, 10, 8, 8]
    expected_ids = ["a", "a", "c", "c"]

    result = regret(data)

    assert all(result["best"].values == expected_best)
    assert all(result["best_id"].values == expected_ids)

    data = pd.DataFrame(
        data={
            "id": ["a", "b", "c"],
            "objective": [10, 9, 8],
        }
    )

    expected_best = [10, 9, 8]
    expected_ids = ["a", "b", "c"]

    result = regret(data)

    assert all(result["best"].values == expected_best)
    assert all(result["best_id"].values == expected_ids)

    data = pd.DataFrame(
        data={
            "id": ["a", "b", "c"],
            "objective": [8, 9, 10],
        }
    )

    expected_best = [8, 8, 8]
    expected_ids = ["a", "a", "a"]

    result = regret(data)

    assert all(result["best"].values == expected_best)
    assert all(result["best_id"].values == expected_ids)

    data = pd.DataFrame(
        data={
            "id": ["a"],
            "objective": [10],
        }
    )

    result = regret(data)

    assert all(result["best"].values == [10])
    assert all(result["best_id"].values == ["a"])


def test_regret_equal():
    """Tests instances where trials share best objectives"""
    data = pd.DataFrame(
        data={
            "id": ["a", "b", "c"],
            "objective": [8, 9, 8],
        }
    )

    expected_best = [8, 8, 8]
    expected_ids = ["a", "a", "a"]

    result = regret(data)

    assert all(result["best"].values == expected_best)
    assert all(result["best_id"].values == expected_ids)

    data = pd.DataFrame(
        data={
            "id": ["a", "b", "c"],
            "objective": [8, 8, 8],
        }
    )

    expected_best = [8, 8, 8]
    expected_ids = ["a", "a", "a"]

    result = regret(data)

    assert all(result["best"].values == expected_best)
    assert all(result["best_id"].values == expected_ids)
