# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.random`."""


from base import ObjectiveStub, TrialStub

from orion.algo.pbt.pbt import Lineage, Lineages, compute_fidelities
from orion.testing.algo import BaseAlgoTests


class TestComputeFidelities:
    def test_base_1(self):
        assert compute_fidelities(10, 10, 20, 1).tolist() == list(
            map(float, range(10, 21))
        )

    def test_other_bases(self):
        assert compute_fidelities(9, 2, 2 ** 10, 2).tolist() == [
            2 ** i for i in range(1, 11)
        ]


class TestPBT(BaseAlgoTests):
    algo_name = "pbt"
    config = {"seed": 123456}


# TestRandomSearch.set_phases([("random", 0, "space.sample")])
