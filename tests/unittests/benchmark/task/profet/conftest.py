from logging import getLogger as get_logger
from pathlib import Path

import numpy as np
import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory

from orion.benchmark.task.profet.profet_task import MetaModelConfig

logger = get_logger(__name__)

REAL_PROFET_DATA_DIR: Path = Path("profet_data")


def is_nonempty_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and bool(list(p.iterdir()))


@pytest.fixture()
def checkpoint_dir(tmp_path: Path):
    return tmp_path


@pytest.fixture(scope="session")
def profet_input_dir(tmp_path_factory):
    """Returns the `input_dir` to use for the Profet algorithm."""
    if REAL_PROFET_DATA_DIR.exists():
        return REAL_PROFET_DATA_DIR
    return tmp_path_factory.mktemp("profet_data")


logger = get_logger(__name__)


@pytest.fixture(autouse=True, params=[True, False], ids=["real_data", "fake_data"])
def mock_load_data(
    monkeypatch: MonkeyPatch, request: SubRequest, tmp_path_factory: TempPathFactory
):
    """Fixture used in all the profet task tests that does the following:

    Parametrizes all tests so they are run with both real (when possible) and fake training data.
    - If the profet data is downloaded, then the tests are run with both real data & fake data.
    - If the profet data is not downloaded, then the tests with real data are skipped, and the tests
      are run with fake data.

    This works by modifying `load_data` so it doesn't attempt to download the data, and instead
    mocks it so that it generates random data with the same shape and basic stats as the real
    datasets (shown above).
    """
    use_real_data: bool = request.param

    if use_real_data and not is_nonempty_dir(REAL_PROFET_DATA_DIR):
        pytest.skip(
            f"Real profet data is not found in dir '{REAL_PROFET_DATA_DIR}', skipping test."
        )

    real_load_data = MetaModelConfig.load_data
    # real_load_data = orion.benchmark.task.profet.model_utils.MetaModelConfig.load_data

    def _load_data(self: MetaModelConfig, path: Path):
        if use_real_data and path == REAL_PROFET_DATA_DIR:
            # Return real datasets.
            logger.info("Testing using the real Profet datasets.")
            return real_load_data(self, REAL_PROFET_DATA_DIR)
        # Generate fake datasets.
        logger.info("Using random data instead of the actual profet training data.")
        x_shape, y_shape, c_shape = self.shapes
        X = np.random.rand(*x_shape)
        min_y = self.y_min
        max_y = self.y_max
        Y = np.random.rand(*y_shape) * (max_y - min_y) + min_y
        min_c = self.c_min
        max_c = self.c_max
        C = np.random.rand(*c_shape) * (max_c - min_c) + min_c
        return X, Y, C

    monkeypatch.setattr(MetaModelConfig, "load_data", _load_data)
