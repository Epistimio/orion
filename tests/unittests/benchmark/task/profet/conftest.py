from logging import getLogger as get_logger
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import orion.benchmark.task.profet.profet_task
import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig
import random
import torch
import numpy as np

logger = get_logger(__name__)

REAL_PROFET_DATA_DIR: Path = Path("profet_data")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_nonempty_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and bool(list(p.iterdir()))


@pytest.fixture(scope="session")
def profet_train_config():
    """ Fixture that provides a configuration object for the Profet algorithm for testing. """
    quick_train_config = MetaModelTrainingConfig(
        num_burnin_steps=10,
        # num_steps=10,
        max_iters=10,
        n_samples_task=20,
    )
    return quick_train_config


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("checkpoint_dir")


@pytest.fixture(scope="session")
def profet_input_dir(tmp_path_factory):
    # Returns the `input_dir` to use for the Profet algorithm.
    if REAL_PROFET_DATA_DIR.exists():
        return REAL_PROFET_DATA_DIR
    return tmp_path_factory.mktemp("profet_data")


logger = get_logger(__name__)
shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {
    "fcnet": ((600, 6), (27, 600), (27, 600)),
    "forrester": ((10, 2), (9, 10), (9, 10)),
    "svm": ((200, 2), (26, 200), (26, 200)),
    "xgboost": ((800, 8), (11, 800), (11, 800)),
}
y_min: Dict[str, float] = {
    "fcnet": 0.0,
    "forrester": -18.049155413936802,
    "svm": 0.0,
    "xgboost": 0.0,
}
y_max: Dict[str, float] = {
    "fcnet": 1.0,
    "forrester": 14718.31848526001,
    "svm": 1.0,
    "xgboost": 3991387.335843141,
}
c_min: Dict[str, float] = {
    "fcnet": 0.0,
    "forrester": -18.049155413936802,
    "svm": 0.0,
    "xgboost": 0.0,
}
c_max: Dict[str, float] = {
    "fcnet": 14718.31848526001,
    "forrester": 14718.31848526001,
    "svm": 697154.4010462761,
    "xgboost": 5485.541382551193,
}


@pytest.fixture(autouse=True, params=[True, False], ids=["real_data", "fake_data"])
def mock_load_data(
    monkeypatch: MonkeyPatch, request: SubRequest, tmp_path_factory: TempPathFactory
):
    """Fixture used in all the profet task tests that, when the real profet data isn't available,
    modifies `load_data` so it doesn't attempt to download the data, and instead mocks it so that it
    generates random data with the same shape and basic stats as the real datasets (shown above).

    NOTE: This fixture is parametrized so that we also run the tests with the fake data, even when
    the real data is available. 
    """
    use_real_data: bool = request.param

    if use_real_data and not is_nonempty_dir(REAL_PROFET_DATA_DIR):
        pytest.skip(
            f"Real profet data is not found in dir '{REAL_PROFET_DATA_DIR}', skipping test."
        )

    real_load_data = orion.benchmark.task.profet.model_utils.load_data

    def _load_data(path: Path, benchmark: str):
        if use_real_data and path == REAL_PROFET_DATA_DIR:
            # Return real datasets.
            logger.info(f"Testing using the real Profet datasets.")
            return real_load_data(REAL_PROFET_DATA_DIR, benchmark=benchmark)
        # Generate fake datasets.
        logger.info(f"Warning: Using random data instead of the actual profet training data.")
        x_shape, y_shape, c_shape = shapes[benchmark]
        X = np.random.rand(*x_shape)
        min_y = y_min[benchmark]
        max_y = y_max[benchmark]
        Y = np.random.rand(*y_shape) * (max_y - min_y) + min_y
        min_c = c_min[benchmark]
        max_c = c_max[benchmark]
        C = np.random.rand(*c_shape) * (max_c - min_c) + min_c
        return X, Y, C

    monkeypatch.setattr(orion.benchmark.task.profet.model_utils, "load_data", _load_data)
    # NOTE: Need to set the item in the globals because it might already have been imported.
    monkeypatch.setitem(globals(), "load_data", _load_data)
