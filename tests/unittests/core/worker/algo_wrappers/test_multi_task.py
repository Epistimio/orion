import pytest
from orion.core.worker.knowledge_base import KnowledgeBase


class DummyKnowledgeBase(KnowledgeBase):
    ...


@pytest.fixture()
def knowledge_base():
    pass


class TestMultiTaskWrapper:
    """Tests for the multi-task wrapper."""

    def test_adds_task_id(self):
        """Test that when an algo is wrapped with the multi-task wrapper, the trials it returns
        with suggest() have an additional 'task-id' value.
        """
