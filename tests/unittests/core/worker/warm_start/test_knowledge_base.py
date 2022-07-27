from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from orion.algo.space import Space
from orion.benchmark.task import RosenBrock
from orion.client import build_experiment
from orion.client.experiment import ExperimentClient
from orion.core.io.database.pickleddb import PickledDB
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.warm_start import KnowledgeBase
from orion.storage.base import BaseStorageProtocol
from orion.storage.legacy import Legacy

# Function to create a space.
_space: Callable[[dict], Space] = SpaceBuilder().build


@pytest.fixture()
def knowledge_base(tmp_path: Path):
    """Fixture that creates a temporary storage with some trials we want, and then passes it
    to the KB.
    """
    db = PickledDB(host=f"{tmp_path}/db.pkl")
    storage = Legacy(database=db, setup=True)
    # TODO: Add experiments useful for tests in here.
    knowledge_base = KnowledgeBase(storage)
    return knowledge_base


class TestKnowledgeBase:
    def test_max_trials_is_respected(self, storage: BaseStorageProtocol):
        """Test that the knowledge base retrieves at most `max_trials` trials in total."""
        n_previous_experiments = 2
        n_trials_to_create = 10
        n_trials_to_fetch = 20

        source_experiments: list[ExperimentClient] = []
        task = RosenBrock()
        assert len(storage.fetch_experiments({})) == 0
        for i in range(n_previous_experiments):
            source_experiment = build_experiment(
                name=f"source_{i}",
                space=task.get_search_space(),
                max_trials=n_trials_to_create,
            )
            source_experiment.workon(task, max_trials=n_trials_to_create)
            source_experiments.append(source_experiment)
        assert len(storage.fetch_experiments({})) == n_previous_experiments
        n_prev_trials = n_trials_to_create * n_previous_experiments

        knowledge_base = KnowledgeBase(storage=storage)

        target_experiment = build_experiment(
            name="foo",
            space={"x": "uniform(0, 1)"},
            debug=True,
        )
        related_trials = knowledge_base.get_related_trials(
            target_experiment,
            max_trials=n_trials_to_fetch,
        )

        assert len(related_trials) == n_previous_experiments
        for i, (exp_config, trials) in enumerate(related_trials):
            # NOTE: There is a slight mismatch between what comes back from the Storage
            # and the configuration property of the actual Experiment object.
            # assert exp_config == source_experiments[i].configuration
            assert exp_config["_id"] == source_experiments[i].configuration["_id"]

            assert len(trials) <= n_trials_to_fetch
        assert sum(len(trials) for _, trials in related_trials) == min(
            n_prev_trials, n_trials_to_fetch
        )

    def test_similarity_fn_is_used(self):
        ...

    def test_num_experiments_property(self):
        ...
