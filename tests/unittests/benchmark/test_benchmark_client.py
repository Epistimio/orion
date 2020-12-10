#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.benchmark_client`."""

import pytest
from orion.core.utils.tests import OrionState
from orion.core.utils.exceptions import NoConfigurationError

from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.assessment import AverageResult, AverageRank
from orion.benchmark.task import RosenBrock, CarromTable


class DummyTask():
    pass


class DummyAssess():
    pass


@pytest.fixture
def algorithms():
    return [{'random': {'seed': 1}}, {'tpe': {'seed': 1}}]


class TestCreateBenchmark():

    def test_create_benchmark(self, algorithms):

        with OrionState():
            bm1 = get_or_create_benchmark('bm00001', algorithms=algorithms,
                                     targets=[{'assess':[AverageResult(2), AverageRank(2)],
                                               'task': [RosenBrock(25, dim=3), CarromTable(20)]}])

            bm2 = get_or_create_benchmark('bm00001')

            assert bm1.configuration == {'name': 'bm00001',
                                         'algorithms': algorithms,
                                         'targets': [
                                             {'assess': [
                                                 {'orion-benchmark-assessment-averageresult-AverageResult': {'task_num': 2}},
                                                 {'orion-benchmark-assessment-averagerank-AverageRank': {'task_num': 2}}],
                                             'task': [
                                                 {'orion-benchmark-task-rosenbrock-RosenBrock': {'dim': 3, 'max_trials': 25}},
                                                 {'orion-benchmark-task-carromtable-CarromTable': {'max_trials': 20}}]}]}

            assert bm1.configuration == bm2.configuration

    def test_create_with_only_name(self):
        with OrionState():
            name = 'bm00001'
            with pytest.raises(NoConfigurationError) as exc:
                get_or_create_benchmark(name)

            assert 'Benchmark {} does not exist in DB'.format(name) in str(exc.value)

    def test_create_with_invalid_algorithms(self):
        with OrionState():
            name = 'bm00001'
            with pytest.raises(NotImplementedError) as exc:
                get_or_create_benchmark(name, algorithms=[{'fake_algorithm': {'seed': 1}}],
                                        targets=[{'assess':[AverageResult(2), AverageRank(2)],
                                                  'task': [RosenBrock(25, dim=3), CarromTable(20)]}])
            assert 'Could not find implementation of BaseAlgorithm' in str(exc.value)

    def test_create_with_invalid_targets(self, algorithms):
        with OrionState():
            name = 'bm00001'
            with pytest.raises(AttributeError) as exc:
                get_or_create_benchmark(name, algorithms=algorithms,
                                        targets=[{'assess':[AverageResult(2)],
                                                  'task': [DummyTask]}])
            print(str(exc.value))

            assert "type object '{}' has no attribute ".format('DummyTask') in str(exc.value)

            with pytest.raises(AttributeError) as exc:
                get_or_create_benchmark(name, algorithms=algorithms,
                                        targets=[{'assess':[DummyAssess],
                                                  'task': [RosenBrock(25, dim=3)]}])
            print(str(exc.value))

            assert "type object '{}' has no attribute ".format('DummyAssess') in str(exc.value)
