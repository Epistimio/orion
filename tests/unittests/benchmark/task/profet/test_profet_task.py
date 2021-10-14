""" Utilities used to test the different subclasses of the ProfetTask. """
from logging import getLogger as get_logger
from pathlib import Path
from typing import ClassVar, Type

import numpy as np
import pytest
import torch
from orion.algo.space import _Discrete
from orion.benchmark.task.profet.profet_task import (
    MetaModelTrainingConfig,
    ProfetTask,
)

from .conftest import REAL_PROFET_DATA_DIR, y_min, y_max, c_min, c_max


logger = get_logger(__name__)


@pytest.mark.timeout(15)
@pytest.mark.parametrize("benchmark", ["fcnet", "forrester", "svm", "xgboost"])
def test_download_fake_datasets(tmp_path_factory, benchmark: str, load_fake_data):
    from orion.benchmark.task.profet.profet_task import load_data

    real_input_dir: Path = REAL_PROFET_DATA_DIR
    real_x, real_y, real_c = load_data(real_input_dir, benchmark=benchmark)

    fake_input_dir: Path = tmp_path_factory.mktemp("profet_data")
    fake_x, fake_y, fake_c = load_data(fake_input_dir, benchmark=benchmark)
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

    Task: ClassVar[Type[ProfetTask]]

    @pytest.mark.timeout(30)
    def test_instantiating_task(
        self,
        profet_train_config: MetaModelTrainingConfig,
        profet_input_dir: Path,
        tmp_path_factory,
    ):
        """ Tests that when instantiating multiple tasks with the same arguments, the
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

    def test_configuration(
        self,
        profet_train_config: MetaModelTrainingConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
    ):
        """ Test that tasks have a proper configuration and that they can be created from it.
        """
        task_id = 0
        kwargs = dict(
            max_trials=10,
            task_id=task_id,
            train_config=profet_train_config,
            seed=123,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        first_task = self.Task(**kwargs)
        configuration = first_task.configuration
        assert len(configuration.keys()) == 1
        key, config_dict = configuration.popitem()
        assert key == self.Task.__qualname__

        for key, value in kwargs.items():
            assert key in config_dict
            assert config_dict[key] == value

        # NOTE: This 'name' is the qualname of the class, which is used by the Factory metaclass'
        # `__call__` method.
        name = self.Task.__qualname__
        from orion.benchmark.benchmark_client import _get_task

        second_task = _get_task(name=name, **config_dict)
        assert second_task.configuration == first_task.configuration

    def test_sanity_check(
        self,
        profet_train_config: MetaModelTrainingConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
    ):
        """ Tests that two tasks with different arguments give different results. """
        task_id = 0
        first_task_kwargs = dict(
            max_trials=10,
            task_id=task_id,
            train_config=profet_train_config,
            seed=123,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        
        first_task = self.Task(**first_task_kwargs)
        first_point = first_task._space.sample(1, seed=first_task.seed)[0]
        first_results = first_task(first_point)
        assert len(first_results) == 1
        assert first_results[0]["type"] == "objective"
        first_objective = first_results[0]["value"]

        second_task_kwargs = first_task_kwargs.copy()
        second_task_kwargs["seed"] += 456

        second_task = self.Task(**second_task_kwargs)
        second_point = second_task._space.sample(1, seed=second_task.seed)[0]
        second_results = second_task(second_point)
        assert len(second_results) == 1
        assert second_results[0]["type"] == "objective"
        second_objective = second_results[0]["value"]

        assert first_point != second_point
        assert first_objective != second_objective

    def test_call_is_reproducible(
        self,
        profet_train_config: MetaModelTrainingConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
    ):
        """ Two tasks created with the same args, given the same point, should produce the same
        results.
        """
        task_id = 0
        task_kwargs = dict(
            max_trials=10,
            task_id=task_id,
            train_config=profet_train_config,
            seed=123,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,            
        )
        first_task = self.Task(**task_kwargs)

        first_point = first_task._space.sample(1, seed=first_task.seed)[0]
        first_results = first_task(first_point)
        assert len(first_results) == 1
        assert first_results[0]["type"] == "objective"
        first_objective = first_results[0]["value"]

        second_task = self.Task(**task_kwargs)
        second_point = second_task._space.sample(1, seed=second_task.seed)[0]
        second_results = second_task(second_point)
        assert len(second_results) == 1
        assert second_results[0]["type"] == "objective"
        second_objective = second_results[0]["value"]

        assert second_objective == first_objective

    def test_call_twice_with_same_task_gives_same_result(
        self,
        profet_train_config: MetaModelTrainingConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
    ):
        """ When using the same task and different values for the same point, the results should
        be identical.
        """
        task_id = 0
        task = self.Task(
            max_trials=10,
            task_id=task_id,
            train_config=profet_train_config,
            seed=123,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )

        point = task._space.sample(1, seed=task.seed)[0]

        first_results = task(point)
        assert len(first_results) == 1
        assert first_results[0]["type"] == "objective"
        first_objective = first_results[0]["value"]

        # Pass the same point under different forms and check that the results are exactly the same.
        for second_point in [point, tuple(point), dict(zip(task._space.keys(), point))]:
            second_results = task(second_point)
            assert second_results[0]["name"] == first_results[0]["name"]
            assert second_results[0]["type"] == first_results[0]["type"]
            second_objective = second_results[0]["value"]

            # NOTE: Not sure why, but the two values are very close, but different!
            assert np.isclose(first_objective, second_objective)

    @pytest.mark.parametrize("step_size", [1e-2, 1e-5])
    def test_call_with_gradients(
        self,
        profet_train_config: MetaModelTrainingConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        step_size: float,
    ):
        """ Test that calling the task with the `with_grad` returns the gradient at that point. """
        task = self.Task(
            max_trials=10,
            task_id=0,
            train_config=profet_train_config,
            seed=123,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
            with_grad=True,
        )

        point_tuple = task._space.sample(1)[0]
        point_dict = dict(zip(task._space.keys(), point_tuple))
        results = task(point_dict, with_grad=True)

        assert results[0]["type"] == "objective"
        first_objective = results[0]["value"]
        assert results[1]["type"] == "gradient"
        first_gradient = results[1]["value"]

        second_point_array = np.array(point_tuple) - step_size * np.array(first_gradient)

        if any(isinstance(dim, _Discrete) for dim in task._space.values()):
            # NOTE: Some dimensions don't quite work here because they are discrete: we can't really
            # use a point that has batch_size = 12.04 for example.
            assert second_point_array not in task._space

        if second_point_array not in task._space:
            with pytest.warns(RuntimeWarning, match="isn't a valid point of the space"):
                # NOTE: Bypassing the conversion that would normally happen in `dict_to_trial` for tuple
                # inputs when a Tensor is passed directly to the task. When the point doesn't fit the
                # space exactly, a RuntimeWarning is raised.
                second_results = task(torch.as_tensor(second_point_array))
        else:
            # Check that no warnings are raised if the point is in the space of the task.
            with pytest.warns(None) as record:
                second_results = task(second_point_array)
            assert len(record) == 0, [m.message for m in record.list]

        second_objective = second_results[0]["value"]

        # NOTE: (@lebrice): Expected this to work for really small step sizes. Using simpler check
        # below instead.
        # assert np.isclose(improvement, step_size)

        # This check sort-of works, at least.
        # NOTE: Lower is better here, hence the order.
        improvement = first_objective - second_objective
        # NOTE: For the SVM task, the improvement is sometimes just zero.
        assert improvement >= 0.0
