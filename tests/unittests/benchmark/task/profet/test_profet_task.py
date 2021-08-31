""" Utilities used to test the different subclasses of the ProfetTask. """
from pathlib import Path
from typing import ClassVar, Type
import pytest

from orion.benchmark.task.base import BaseTask
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig, ProfetTask, download_data, load_data
from typing import Dict, Tuple
import numpy as np

from logging import getLogger as get_logger
from .conftest import REAL_PROFET_DATA_DIR, requires_profet_data

logger = get_logger(__name__)
shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]] = {
    "fcnet": ((600, 6), (27, 600), (27, 600)),
    "forrester": ((10, 1), (9, 10), (9, 10)),
    "svm": ((200, 2), (26, 200), (26, 200)),
    "xgboost": ((800, 8), (11, 800), (11, 800)),
}
y_min: Dict[str, float] = {
    "fcnet": 0.,
    "forrester": -18.049155413936802,
    "svm": 0.,
    "xgboost": 0.,
}
y_max: Dict[str, float] = {
    "fcnet": 1.,
    "forrester": 14718.31848526001,
    "svm": 1.,
    "xgboost": 3991387.335843141,
}
c_min: Dict[str, float] = {
    "fcnet": 0.,
    "forrester": -18.049155413936802,
    "svm": 0.,
    "xgboost": 0.,
}
c_max: Dict[str, float] = {
    "fcnet": 14718.31848526001,
    "forrester": 14718.31848526001,
    "svm": 697154.4010462761,
    "xgboost": 5485.541382551193,
}

@pytest.fixture(autouse=True)
def load_fake_data(monkeypatch, tmp_path_factory):
    """ Fixture that prevents attempts to download the true Profet datasets, and instead generates
    random data with the same shape.
    """
    

    import orion.benchmark.task.profet.profet_task
    real_load_data = orion.benchmark.task.profet.profet_task.load_data

    def _load_data(path: Path, benchmark: str):
        if path == REAL_PROFET_DATA_DIR and REAL_PROFET_DATA_DIR.exists():
            # Return real datasets.
            logger.info(f"Testing using the real Profet datasets.")
            return real_load_data(path, benchmark=benchmark)
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

    monkeypatch.setattr(orion.benchmark.task.profet.profet_task, "load_data", _load_data)
    monkeypatch.setitem(globals(), "load_data", _load_data)


@pytest.mark.timeout(10)
@requires_profet_data
@pytest.mark.parametrize("benchmark", ["fcnet", "forrester", "svm", "xgboost"])
def test_download_fake_datasets(tmp_path_factory, benchmark: str, load_fake_data):
    # TODO: Downloading the data takes a VERY long time, even though the datasets are relatively
    # small! Would it be alright to store those datasets somewhere?
    real_input_dir: Path = REAL_PROFET_DATA_DIR
    real_x, real_y, real_c = load_data(real_input_dir, benchmark=benchmark)

    fake_input_dir: Path = tmp_path_factory.mktemp("profet_data")
    fake_x, fake_y, fake_c = load_data(fake_input_dir, benchmark=benchmark)
    # assert False, (x.dtype, y.dtype, c.dtype)
    # assert False, (x.dtype, y.dtype, c.dtype)
    assert real_x.shape == fake_x.shape
    assert real_y.shape == fake_y.shape
    assert real_c.shape == fake_c.shape

    assert real_x.dtype == fake_x.dtype
    assert real_y.dtype == fake_y.dtype
    assert real_c.dtype == fake_c.dtype

    assert 0 <= real_x.min() and 0 <= fake_x.min()
    assert real_x.max() <= 1 and fake_x.max() <= 1
    
    min_y, max_y = y_min[benchmark], y_max[benchmark]
    assert min_y <= real_y.min() and min_y <= fake_y.min()
    assert real_y.max() <= max_y and fake_y.max() <= max_y

    min_c, max_c = c_min[benchmark], c_max[benchmark]
    assert min_c <= real_c.min() and min_c <= fake_c.min()
    assert real_c.max() <= max_c and fake_c.max() <= max_c


class ProfetTaskTests:
    """ Base class for testing Profet tasks. """

    # TODO: What would be a good set of tests for the profet tasks?

    Task: ClassVar[Type[ProfetTask]]

    @pytest.mark.timeout(30)
    def test_instantiating_task(
        self,
        profet_train_config: MetaModelTrainingConfig,
        profet_input_dir: Path,
        tmp_path_factory,
    ):
        """ TODO: Test that when instantiating multiple tasks with the same arguments, the
        meta-model is trained only once.
        """
        max_trials = 123
        task_id = 0
        seed = 123
        
        checkpoint_dir: Path = tmp_path_factory.mktemp("checkpoints")
        # Directory should be empty.
        assert len(list(checkpoint_dir.iterdir())) == 0
        
        task = self.Task(
            max_trials=max_trials,
            task_id=task_id,
            train_config=profet_train_config,
            seed=seed,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        assert task.max_trials == max_trials
        assert task.seed == seed
        assert task.task_id == task_id
        
        # Directory should have one file (the trained model).
        assert len(list(checkpoint_dir.iterdir())) == 1

        task = self.Task(
            max_trials=max_trials,
            task_id=task_id,
            train_config=profet_train_config,
            seed=seed,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        assert task.max_trials == max_trials
        assert task.seed == seed
        assert task.task_id == task_id

        # Directory should have one file (the trained model).
        assert len(list(checkpoint_dir.iterdir())) == 1

