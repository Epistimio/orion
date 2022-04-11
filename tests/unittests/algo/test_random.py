#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.random`."""
from __future__ import annotations

from typing import ClassVar

import numpy
import pytest

from orion.algo.random import Random
from orion.algo.space import Integer, Real, Space
from orion.testing.algo import BaseAlgoTests, TestPhase


class TestRandomSearch(BaseAlgoTests):
    algo_name = "random"
    config = {"seed": 123456}
    phases: ClassVar[list[TestPhase]] = [TestPhase("random", 0, "space.sample")]
