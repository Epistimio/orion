from pathlib import Path
import pytest
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig, NAMES

REAL_PROFET_DATA_DIR: Path = Path("profet_data")

requires_profet_data = pytest.mark.skipif(
    not REAL_PROFET_DATA_DIR.exists(),
    reason="Need to have the real profet data downloaded to run this test.",
)


@pytest.fixture(scope="session")
def profet_train_config():
    """ Fixture that provides a configuration object for the Profet algorithm for testing. """
    # TODO: Figure out a good set of values that makes the training of the meta-model really fast,
    # or use a monkeypatch fixture to make training super quick to run somehow.
    quick_train_config = MetaModelTrainingConfig(
        num_burnin_steps=10,
        # num_steps=10,
        max_iters=10,
        n_samples_task=20,
    )
    return quick_train_config


@pytest.fixture(scope="session")
def profet_checkpoint_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("checkpoint_dir")


@pytest.fixture(scope="session")
def profet_input_dir(tmp_path_factory):
    # TODO: Return the `input_dir` to use for the Profet algorithm.
    return "profet_data"
