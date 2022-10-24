#!/usr/bin/env python
"""Tests for :mod:`orion.benchmark.task`."""

from orion.benchmark.task import Branin, CarromTable, EggHolder, RosenBrock


class TestBranin:
    """Test benchmark task branin"""

    def test_creation(self):
        """Test creation"""
        task = Branin(2)
        assert task.max_trials == 2
        assert task.configuration == {"Branin": {"max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        task = Branin(2)

        assert callable(task)

        objectives = task([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        task = Branin(2)

        assert task.get_search_space() == {"x": "uniform(0, 1, shape=2, precision=10)"}


class TestCarromTable:
    """Test benchmark task CarromTable"""

    def test_creation(self):
        """Test creation"""
        task = CarromTable(2)
        assert task.max_trials == 2
        assert task.configuration == {"CarromTable": {"max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        task = CarromTable(2)

        assert callable(task)

        objectives = task([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        task = CarromTable(2)

        assert task.get_search_space() == {"x": "uniform(-10, 10, shape=2)"}


class TestEggHolder:
    """Test benchmark task EggHolder"""

    def test_creation(self):
        """Test creation"""
        task = EggHolder(max_trials=2, dim=3)
        assert task.max_trials == 2
        assert task.configuration == {"EggHolder": {"dim": 3, "max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        task = EggHolder(2)

        assert callable(task)

        objectives = task([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        task = EggHolder(2)

        assert task.get_search_space() == {"x": "uniform(-512, 512, shape=2)"}


class TestRosenBrock:
    """Test benchmark task RosenBrock"""

    def test_creation(self):
        """Test creation"""
        task = RosenBrock(max_trials=2, dim=3)
        assert task.max_trials == 2
        assert task.configuration == {"RosenBrock": {"dim": 3, "max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        task = RosenBrock(2)

        assert callable(task)

        objectives = task([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        task = RosenBrock(2)
        assert task.get_search_space() == {"x": "uniform(-5, 10, shape=2)"}
