from __future__ import annotations

import copy
import pytest

from orion.algo.space import Space
from orion.core.worker.warm_start import KnowledgeBase
from typing import Any, Callable
from orion.core.io.space_builder import SpaceBuilder
from orion.client import ExperimentClient
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.testing.state import OrionState

# Function to create a space.
space: Callable[[dict], Space] = SpaceBuilder().build

previous_spaces: list[Space] = [
    space({"x": "uniform(0, 5)"}),
    space({"x": "uniform(1, 6)"}),
    space({"x": "uniform(2, 7)"}),
    space({"x": "uniform(3, 8)"}),
    space({"x": "uniform(4, 9)"}),
]
target_space = space({"x": "uniform(0, 10)"})


@pytest.fixture()
def knowledge_base():
    pass


class TestMultiTaskWrapper:
    """Tests for the multi-task wrapper."""

    def test_adds_task_id(self):
        """Test that when an algo is wrapped with the multi-task wrapper, the trials it returns
        with suggest() have an additional 'task-id' value.
        """
