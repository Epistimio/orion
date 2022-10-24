#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.algo.random`."""
from __future__ import annotations

from typing import ClassVar

from orion.algo.random import Random
from orion.testing.algo import BaseAlgoTests


class TestRandomSearch(BaseAlgoTests):
    """Tests for the Random algorithm."""

    algo_type: ClassVar[type[Random]] = Random
    config: ClassVar[dict] = {"seed": 123456}
