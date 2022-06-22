from __future__ import annotations

from typing import Callable

import pytest

from orion.algo.space import Space
from orion.core.io.space_builder import SpaceBuilder

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
