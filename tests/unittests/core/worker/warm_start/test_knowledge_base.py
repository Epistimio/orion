from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import pytest

from orion.algo.space import Space
from orion.core.io.database.pickleddb import PickledDB
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start import KnowledgeBase
from orion.storage.legacy import Legacy

# Function to create a space.
space: Callable[[dict], Space] = SpaceBuilder().build


def add_result(trial: Trial, objective: float) -> Trial:
    """Add `objective` as the result of `trial`. Returns a new Trial object."""
    new_trial = copy.deepcopy(trial)
    new_trial.status = "completed"
    new_trial.results.append(
        Trial.Result(name="objective", type="objective", value=objective)
    )
    return new_trial


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
    def test_max_trials_is_respected(self):
        ...

    def test_similarity_fn_is_used(self):
        ...

    def test_num_experiments_property(self):
        ...
