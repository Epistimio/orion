""" Tests for the KnowledgeBase. """
from __future__ import annotations

from pathlib import Path

import pytest

from orion.benchmark.task import RosenBrock
from orion.client import build_experiment
from orion.client.experiment import ExperimentClient
from orion.core.io.database.pickleddb import PickledDB
from orion.core.worker.experiment_config import ExperimentConfig
from orion.core.worker.warm_start import KnowledgeBase
from orion.storage.base import BaseStorageProtocol
from orion.storage.legacy import Legacy


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


def create_experiments(
    n_previous_experiments: int,
    n_trials_per_experiment: int,
    storage: BaseStorageProtocol,
    prefix: str = "source",
) -> list[ExperimentClient]:
    """Fixture that creates a set of experiments, and adds them to the given storage object."""
    # NOTE: The experiments are implicitly added to the Storage object, due to the storage being a
    # singleton.
    task = RosenBrock()
    source_experiments: list[ExperimentClient] = []
    assert len(storage.fetch_experiments({})) == 0

    for i in range(n_previous_experiments):
        source_experiment = build_experiment(
            name=f"{prefix}_{i}",
            space=task.get_search_space(),
            max_trials=n_trials_per_experiment,
            storage=storage,
        )
        source_experiment.workon(task, max_trials=n_trials_per_experiment)
        source_experiments.append(source_experiment)
    assert len(storage.fetch_experiments({})) == n_previous_experiments
    return source_experiments


class TestKnowledgeBase:
    """Tests for the KnowledgeBase class."""

    def test_empty_kb(self, storage: BaseStorageProtocol):
        knowledge_base = KnowledgeBase(storage)
        assert knowledge_base.n_stored_experiments == 0
        target_experiment = build_experiment(
            name="foo",
            space={"x": "uniform(0, 1)"},
            debug=True,
        )
        assert not knowledge_base.get_related_trials(target_experiment)
        # TODO: Uncomment this once https://github.com/Epistimio/orion/pull/942 is merged.
        # assert knowledge_base.n_stored_experiments == 0

    def test_max_trials_is_respected(
        self, storage: BaseStorageProtocol, tmp_path: Path
    ):
        """Test that the knowledge base retrieves at most `max_trials` trials in total."""
        n_previous_experiments = 2
        n_trials_per_experiment = 10
        max_trials = 15

        n_trials_in_kb = n_previous_experiments * n_trials_per_experiment

        source_experiments = create_experiments(
            n_previous_experiments=n_previous_experiments,
            n_trials_per_experiment=n_trials_per_experiment,
            storage=storage,
        )
        kb = KnowledgeBase(storage)

        target_experiment = build_experiment(
            name="foo",
            space={"x": "uniform(0, 1)"},
            storage={
                "type": "legacy",
                "database": {"type": "pickleddb", "host": f"{tmp_path}/foobar.pkl"},
            },
        )

        related_trials = kb.get_related_trials(
            target_experiment,
            max_trials=max_trials,
        )

        assert len(related_trials) == n_previous_experiments
        for i, (exp_config, trials) in enumerate(related_trials):
            # NOTE: There is a slight mismatch between what comes back from the Storage
            # and the configuration property of the actual Experiment object.
            # For instance, the `working_dir` property goes from None to "", and such.
            # Hence we can't do the full comparison like this:
            # assert exp_config == source_experiments[i].configuration

            assert exp_config["_id"] == source_experiments[i].configuration["_id"]
            assert exp_config["name"] == source_experiments[i].configuration["name"]

        total_fetched_trials = sum(len(trials) for _, trials in related_trials)
        assert total_fetched_trials == min(n_trials_in_kb, max_trials)

    @pytest.mark.parametrize("max_trials", [10, None])
    def test_similarity_fn_is_used(
        self, storage: BaseStorageProtocol, tmp_path: Path, max_trials: int | None
    ):
        n_previous_experiments = 2
        n_trials_per_experiment = 10

        n_trials_in_kb = n_previous_experiments * n_trials_per_experiment
        source_experiments = create_experiments(
            n_previous_experiments=n_previous_experiments,
            n_trials_per_experiment=n_trials_per_experiment,
            storage=storage,
        )

        def _similarity_metric(
            exp_a: ExperimentConfig, exp_b: ExperimentConfig
        ) -> float:
            """Some dummy similarity metric.

            We have experiment names like "source_{i}" and "target_{j}", so the similarity is just
            defined as 1 - abs(i - j).
            """
            a_name = exp_a["name"]
            b_name = exp_b["name"]
            _, _, a_suffix = a_name.partition("_")
            _, _, b_suffix = b_name.partition("_")
            return 1 - abs(int(a_suffix) - int(b_suffix))

        kb = KnowledgeBase(storage, similarity_metric=_similarity_metric)

        # NOTE: Here we use a number higher than the number of experiments in the KB, so that we
        # expect to get the experiments in the reverse order in which they were created.
        target_experiment = build_experiment(
            name="target_100",
            space={"x": "uniform(0, 1)"},
            storage={
                "type": "legacy",
                "database": {"type": "pickleddb", "host": f"{tmp_path}/foobar.pkl"},
            },
        )

        related_trials = kb.get_related_trials(
            target_experiment,
            max_trials=max_trials,
        )
        assert len(related_trials) == n_previous_experiments
        for i, (experiment, trials) in enumerate(related_trials):
            # Check that the experiments are given with a name like "source_{i}" where i is in
            # decreasing order from `n_previous_experiments-1` down to 0.
            assert experiment["name"] == f"source_{n_previous_experiments - 1 - i}"

    @pytest.mark.parametrize("n_previous_experiments", [0, 1, 2])
    def test_num_experiments_property(
        self, storage: BaseStorageProtocol, n_previous_experiments: int
    ):
        create_experiments(
            storage=storage,
            n_previous_experiments=n_previous_experiments,
            n_trials_per_experiment=5,
        )
        kb = KnowledgeBase(storage)
        assert kb.n_stored_experiments == n_previous_experiments
