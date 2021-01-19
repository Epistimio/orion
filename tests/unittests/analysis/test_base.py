# -*- coding: utf-8 -*-
"""Tests :func:`orion.analysis.base`"""
import numpy
import pandas as pd
import pytest

from orion.analysis.base import average, ranking


class TestAverage:
    def test_accept_empty(self):
        """Tests an empty dataframe is returned if you give an empty dataframe"""
        empty_frame = pd.DataFrame()
        result = average(empty_frame)

        assert result.empty
        assert result.equals(empty_frame)

        empty_frame = pd.DataFrame(columns=["order", "best"])
        result = average(empty_frame)

        assert result.empty
        assert "best_mean" not in result.columns
        assert "best_var" not in result.columns
        assert result.equals(empty_frame)

    def test_parameter_not_modified(self):
        """Tests the original dataframe is not modified"""
        data = pd.DataFrame(
            data={"order": ["a", "b", "c", "d"], "best": [0.1, 0.2, 0.3, 0.5]}
        )

        result = average(data)

        assert data.columns.tolist() == ["order", "best"]
        assert result.columns.tolist() == ["order", "best_mean"]

    def test_average_statistic(self):
        """Test that the average is correctly computed"""
        data = pd.DataFrame(
            data={
                "id": [0, 0, 0, 0, 1, 1, 1, 1],
                "order": [0, 1, 2, 3, 0, 1, 2, 3],
                "best": [0.1, 0.2, 0.3, 0.5, 0.3, 0.2, 0.5, 0.6],
            }
        )

        result = average(data)

        ref = (data["best"][:4].to_numpy() + data["best"][4:].to_numpy()) / 2
        numpy.testing.assert_equal(result["best_mean"].to_numpy(), ref)

    def test_variance_statistic(self):
        """Test that the variance is correctly computed"""
        data = pd.DataFrame(
            data={
                "id": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                "order": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                "best": [0.1, 0.2, 0.3, 0.5, 0.3, 0.2, 0.5, 0.6, 0.3, 0.4, 0.1, 0.3],
            }
        )

        result = average(data, return_var=True)

        ref = data["best"].to_numpy().reshape((3, 4)).var(axis=0, ddof=1)

        numpy.testing.assert_allclose(
            result["best_var"].to_numpy(),
            ref,
        )

    def test_unbalanced_groups(self):
        """Test that the mean is correctly computed on unbalanced groups"""
        data = pd.DataFrame(
            data={
                "id": [0, 0, 0, 0, 1, 1, 1, 2, 2],
                "order": [0, 1, 2, 3, 0, 1, 2, 0, 1],
                "best": [0.1, 0.2, 0.3, 0.5, 0.3, 0.2, 0.5, 0.3, 0.4],
            }
        )

        result = average(data, return_var=True)

        assert (
            result["best_mean"][0]
            == (data["best"][0] + data["best"][4] + data["best"][7]) / 3
        )

        assert (
            result["best_mean"][1]
            == (data["best"][1] + data["best"][5] + data["best"][8]) / 3
        )

        assert result["best_mean"][2] == (data["best"][2] + data["best"][6]) / 2

        assert result["best_mean"][3] == data["best"][3]

    def test_custom_group_by(self):
        """Test using custom group_by argument"""
        data = pd.DataFrame(
            data={
                "step": ["a", "b", "c", "d"],
                "best": [0.1, 0.2, 0.3, 0.5],
            }
        )

        result = average(data, group_by="step")

        assert result.columns.tolist() == ["step", "best_mean"]

    def test_custom_key(self):
        """Test using custom key argument"""
        data = pd.DataFrame(
            data={
                "order": ["a", "b", "c", "d"],
                "objective": [0.1, 0.2, 0.3, 0.5],
            }
        )

        result = average(data, key="objective")

        assert result.columns.tolist() == ["order", "objective_mean"]


class TestRanking:
    def test_accept_empty(self):
        """Tests an empty dataframe is returned if you give an empty dataframe"""
        empty_frame = pd.DataFrame()
        result = ranking(empty_frame)

        assert result.empty
        assert result.equals(empty_frame)

        empty_frame = pd.DataFrame(columns=["order", "best"])
        result = ranking(empty_frame)

        assert result.empty
        assert "rank" not in result.columns
        assert result.equals(empty_frame)

    def test_parameter_not_modified(self):
        """Tests the original dataframe is not modified"""
        data = pd.DataFrame(
            data={"order": ["a", "b", "c", "d"], "best": [0.1, 0.2, 0.3, 0.5]}
        )

        result = ranking(data)

        assert data.columns.tolist() == ["order", "best"]
        assert result.columns.tolist() == ["order", "best", "rank"]

    def test_ranking(self):
        """Test that the average is correctly computed"""
        data = pd.DataFrame(
            data={
                "id": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                "order": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                "best": [0.1, 0.2, 0.3, 0.5, 0.3, 0.2, 0.5, 0.6, 0.3, 0.4, 0.1, 0.3],
            }
        )
        result = ranking(data)

        ref = data["best"].to_numpy().reshape((3, 4))

        assert result["rank"].tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0]

    def test_unbalanced_groups(self):
        """Test that the mean is correctly computed on unbalanced groups"""
        data = pd.DataFrame(
            data={
                "id": [0, 0, 0, 0, 1, 1, 1, 2, 2],
                "order": [0, 1, 2, 3, 0, 1, 2, 0, 1],
                "best": [0.1, 0.7, 0.6, 0.5, 0.4, 0.2, 0.5, 0.3, 0.1],
            }
        )

        result = ranking(data)

        assert result["rank"].tolist() == [0, 2, 1, 0, 2, 1, 0, 1, 0]

    def test_custom_group_by(self):
        """Test using custom group_by argument"""
        data = pd.DataFrame(
            data={
                "step": ["a", "b", "c", "d"],
                "best": [0.1, 0.2, 0.3, 0.5],
            }
        )

        result = ranking(data, group_by="step")

        assert result.columns.tolist() == ["step", "best", "rank"]

    def test_custom_key(self):
        """Test using custom key argument"""
        data = pd.DataFrame(
            data={
                "order": ["a", "b", "c", "d"],
                "objective": [0.1, 0.2, 0.3, 0.5],
            }
        )

        result = ranking(data, key="objective")

        assert result.columns.tolist() == ["order", "objective", "rank"]
