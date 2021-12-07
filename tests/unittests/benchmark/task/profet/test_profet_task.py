""" Utilities used to test the different subclasses of the ProfetTask. """
from collections import defaultdict
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Type

import numpy as np
import pytest
import torch
from dataclasses import asdict, replace
from orion.algo.space import _Discrete, Dimension, Space
from orion.benchmark.task.profet.profet_task import (
    MetaModelConfig,
    ProfetTask,
)

from .conftest import (
    REAL_PROFET_DATA_DIR,
    y_min,
    y_max,
    c_min,
    c_max,
    is_nonempty_dir,
    seed_everything,
)

logger = get_logger(__name__)


class ProfetTaskTests:
    """ Base class for testing Profet tasks. """

    Task: ClassVar[Type[ProfetTask]]

    @pytest.fixture()
    def profet_train_config(self):
        """ Fixture that provides a configuration object for the Profet algorithm for testing. """
        quick_train_config = self.Task.ModelConfig(
            task_id=0,
            seed=123,
            num_burnin_steps=10,
            # num_steps=10,
            max_iters=10,
            n_samples_task=20,
        )
        return quick_train_config

    @pytest.mark.skipif(
        is_nonempty_dir(REAL_PROFET_DATA_DIR),
        reason="Need to *not* have the real profet data downloaded to run this test.",
    )
    @pytest.mark.timeout(1)
    def test_mock_load_data_fixture_when_data_isnt_available(self, tmp_path_factory):
        """Test to make sure that the fixture we use to create fake profet data works correctly.
        
        The `mock_load_data` fixture should be automatically used if we don't have the real data
        downloaded.
        """
        fake_input_dir: Path = tmp_path_factory.mktemp("profet_data")

        # We expect the `load_data` function to NOT attempt to download the dataset, hence the timeout.
        fake_x, fake_y, fake_c = self.Task.ModelConfig().load_data(fake_input_dir)
        # NOTE: This might look a bit weird, but it's consistent for each of the real datasets.
        assert fake_x.shape[0] == fake_y.shape[1] == fake_c.shape[1]

    @pytest.mark.skipif(
        not is_nonempty_dir(REAL_PROFET_DATA_DIR),
        reason="Real profet data is required.",
    )
    @pytest.mark.timeout(15)
    def test_mock_load_data_fixture_when_real_data_available(
        self, tmp_path_factory, mock_load_data
    ):
        """ Test that the mock_load_data fixture returns real data when the input dir is non-empty,
        and that the means, dtypes, shapes, etc match between the real and fake data.
        """
        # NOTE: Need to re-import, because the `mock_load_data` fixture modifies that function in the
        # module. If we imported the function at the top of this test module, we would get the original
        # version.

        model_config = self.Task.ModelConfig()

        real_input_dir: Path = REAL_PROFET_DATA_DIR
        real_x, real_y, real_c = model_config.load_data(real_input_dir)

        fake_input_dir: Path = tmp_path_factory.mktemp("profet_data")
        fake_x, fake_y, fake_c = model_config.load_data(fake_input_dir)
        assert real_x.shape == fake_x.shape
        assert real_y.shape == fake_y.shape
        assert real_c.shape == fake_c.shape

        assert real_x.dtype == fake_x.dtype
        assert real_y.dtype == fake_y.dtype
        assert real_c.dtype == fake_c.dtype

        assert 0 <= real_x.min() and 0 <= fake_x.min()
        assert real_x.max() <= 1 and fake_x.max() <= 1

        min_y, max_y = y_min[model_config.benchmark], y_max[model_config.benchmark]
        assert min_y <= real_y.min() and min_y <= fake_y.min()
        assert real_y.max() <= max_y and fake_y.max() <= max_y

        min_c, max_c = c_min[model_config.benchmark], c_max[model_config.benchmark]
        assert min_c <= real_c.min() and min_c <= fake_c.min()
        assert real_c.max() <= max_c and fake_c.max() <= max_c

    @pytest.mark.timeout(30)
    def test_attributes(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        tmp_path_factory,
    ):
        """ Simple test: Check that a task can be created and that its attributes are used and set
        correctly.
        """
        max_trials = 123
        checkpoint_dir: Path = tmp_path_factory.mktemp("checkpoints")
        task = self.Task(
            max_trials=max_trials,
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        assert task.max_trials == max_trials
        # NOTE: The class attributes might differ per task, but that's ok.
        assert isinstance(task.model_config, self.Task.ModelConfig)
        assert asdict(task.model_config) == asdict(profet_train_config)

    @pytest.mark.timeout(30)
    def test_instantiating_task(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        tmp_path_factory,
        monkeypatch,
    ):
        """ Tests that when instantiating multiple tasks with the same arguments, the
        meta-model is trained only once.
        """

        checkpoint_dir: Path = tmp_path_factory.mktemp("checkpoints")
        # Directory should be empty.
        assert len(list(checkpoint_dir.iterdir())) == 0

        MetaModelConfig.get_task_network.__call_counts = 0

        calls: Dict[Callable, int] = {}
        
        def count_calls(f: Callable):
            calls[f] = 0
            def _counted_calls(*args, **kwargs) -> Any:
                calls[f] += 1
                return f(*args, **kwargs)
            return _counted_calls

        _get_task_network = profet_train_config.get_task_network
        _load_task_network = profet_train_config.load_task_network

        monkeypatch.setattr(profet_train_config, "get_task_network", count_calls(profet_train_config.get_task_network))
        monkeypatch.setattr(profet_train_config, "load_task_network", count_calls(profet_train_config.load_task_network))

        assert calls[_get_task_network] == 0
        assert calls[_load_task_network] == 0

        task = self.Task(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )

        assert calls[_get_task_network] == 1
        assert calls[_load_task_network] == 0


        # Directory should have one file (the trained model checkpoint).
        assert len(list(checkpoint_dir.iterdir())) == 1

        task = self.Task(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )

        assert calls[_get_task_network] == 1
        assert calls[_load_task_network] == 1

        # Directory should *still* only have one file (the trained model checkpoint).
        assert len(list(checkpoint_dir.iterdir())) == 1

    @pytest.mark.parametrize("with_grad", [True, False])
    @pytest.mark.parametrize(
        "device_str", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    def test_configuration(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        with_grad: bool,
        device_str: str,
    ):
        """ Test that tasks have a proper configuration and that they can be created from it.
        """
        max_trials = 10
        kwargs = dict(
            max_trials=max_trials,
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
            device=device_str,
            with_grad=with_grad,
        )
        first_task = self.Task(**kwargs)

        assert first_task.configuration == {
            self.Task.__qualname__: dict(
                max_trials=max_trials,
                model_config=asdict(profet_train_config),
                input_dir=str(profet_input_dir),
                checkpoint_dir=str(checkpoint_dir),
                device=device_str,
                with_grad=with_grad,
            )
        }
        from orion.benchmark.benchmark_client import _get_task

        # NOTE: This 'name' is the qualname of the class, which is used by the Factory metaclass'
        # `__call__` method.
        name, config_dict = first_task.configuration.popitem()
        second_task = _get_task(name=name, **config_dict)
        assert second_task.configuration == first_task.configuration

    def test_sample_single_params(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
    ):
        """ Test the `sample` method. """
        task = self.Task(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        hparam_dict = task.sample()
        assert isinstance(hparam_dict, dict)
        assert hparam_dict.keys() == task.get_search_space().keys()
        # BUG?: Space __contains__ doesn't work with a dict?
        # assert hparam_dict in task.space
        assert hparam_dict.values() in task.space

    @pytest.mark.parametrize("n", range(3))
    def test_sample_multiple_params(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        n: int,
    ):
        """ Test the `sample` method. """
        task = self.Task(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        hparam_dicts = task.sample(n)
        assert isinstance(hparam_dicts, list)
        assert len(hparam_dicts) == n

        assert all(
            hparam_dicts[i] != hparam_dicts[j]
            for i in range(n)
            for j in set(range(n)) - {i}
        )

        for hparam_dict in hparam_dicts:
            assert isinstance(hparam_dict, dict)
            assert hparam_dict.keys() == task.get_search_space().keys()
            # NOTE: space doesn't accept dicts:
            assert hparam_dict.values() in task.space

    def test_space_attribute(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
    ):
        """
        BUG: Space.keys() returns *sorted* keys that don't match the ordering in
        `get_search_space`, so the values returned by any space using `space.sample()` are not
        consistent with the ordering of the returned dict from `get_search_space` of the tasks.

        This wouldn't be a big deal if the samples from `space.sample()` were dictionaries, but
        since they are tuples, there is no way of knowing what the ordering of the keys and values
        are.

        Dicts are sorted since python 3.7, there's no need to use OrderedDict anymore, and getting
        things in a different order than you declared them is just confusing.

        Doing this, for example, would be a very subtle bug:

        ```
        hparam_tuple = task.space.sample()
        hparam = dict(zip(task.get_search_space(), hparam_tuple))
        ```
        This hasn't been a problem for the other tasks because they have a single dimension `x`.

        This is why I'm adding a `sample` method on the task itself, so that I don't have to reach
        into the task to get the space in order to create dicts, and so the ordering isn't a
        problem anymore.
        """
        task = self.Task(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        assert isinstance(task.space, Space)
        # BUG mentioned above:
        # assert task.space.keys() == task.get_search_space().keys()

        # Check that all keys are present, no matter the ordering.
        assert set(task.space.keys()) == set(task.get_search_space().keys())

        # NOTE: Can't check the strings directly, since the value isn't always displayed the same
        # way, for example: "1e-6" vs "1e-06". Changing the value in the prior string would be bad.
        # space_string_dict = task.get_search_space()
        # dimension: Dimension
        # for dim_name, dimension in task.space.items():
        #     assert space_string_dict[dim_name] == dimension.get_prior_string()

    def test_sanity_check(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
    ):
        """ Tests that two tasks with different arguments give different results. """
        first_task_kwargs = dict(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )

        first_task = self.Task(**first_task_kwargs)
        first_point = first_task.sample()

        first_results = first_task(first_point)
        assert len(first_results) == 1
        assert first_results[0]["type"] == "objective"
        first_objective = first_results[0]["value"]

        second_task_model_config = replace(
            profet_train_config, seed=profet_train_config.seed + 456
        )
        second_task_kwargs = first_task_kwargs.copy()
        second_task_kwargs["model_config"] = second_task_model_config

        second_task = self.Task(**second_task_kwargs)
        second_point = second_task.sample()
        assert second_point != first_point

        second_results = second_task(second_point)
        assert second_results != first_results

        assert len(second_results) == 1
        assert second_results[0]["type"] == "objective"
        second_objective = second_results[0]["value"]
        assert first_objective != second_objective

    @pytest.mark.parametrize("use_same_model", [True, False])
    def test_call_is_reproducible(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        tmp_path_factory,
        use_same_model: bool,
    ):
        """ Tests for the seeding of the model.
        When creating two tasks with the same parameters & seed:
        - When creating both models from scratch, they should give identical outputs for the same
          input.
        - If the model is created once and loaded in the second task, it should also be the same
          (give the same output for the same input).
        """
        task_kwargs = dict(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        task_a = self.Task(**task_kwargs)
        if use_same_model:
            # Use the same checkpoint directory, meaning that model for task #2 will be loaded from
            # the checkpoint created by task #1.
            task_b = self.Task(**task_kwargs)
        else:
            # NOTE: We can also check if the training is seeded properly by using a different
            # checkpoint directory, which forces the task to re-create a new model using the same
            # training configuration, rather than load the same one as task_a.
            task_b_kwargs = task_kwargs.copy()
            task_b_kwargs["checkpoint_dir"] = tmp_path_factory.mktemp(
                "other_checkpoint_dir"
            )
            task_b = self.Task(**task_b_kwargs)

        point_a = task_a.sample()
        point_b = task_b.sample()

        assert point_a == point_b
        # NOTE: The forward pass samples from a distribution, therefore the values might be
        # different when calling the same task twice on the same input.
        # However, given two different task instances (with the same seed) they should give the same
        # exact sample.
        result_a = task_a(point_a)
        result_b = task_b(point_b)

        assert len(result_a) == 1
        assert result_a[0]["type"] == "objective"
        assert result_a == result_b
        objective_a = result_a[0]["value"]
        objective_b = result_b[0]["value"]
        assert objective_a == objective_b

    @pytest.mark.parametrize("seed", [123, 456])
    def test_call_twice_with_same_task_gives_same_result(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        seed: int,
    ):
        """ When using the same task and the same point twice, the results should be identical.
        """
        model_config = replace(profet_train_config, seed=seed)
        task = self.Task(
            model_config=model_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        assert task.model_config.seed == seed
        point = task.sample()

        first_results = task(point)
        assert len(first_results) == 1
        assert first_results[0]["type"] == "objective"

        # Pass the same point under different forms and check that the results are exactly the same.
        second_results = task(point)
        assert first_results == second_results

    @pytest.mark.parametrize("step_size", [1e-2, 1e-4])
    def test_call_with_gradients(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        step_size: float,
    ):
        """ Test that calling the task with the `with_grad` returns the gradient at that point. """
        task = self.Task(
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
            model_config=profet_train_config,
            with_grad=True,
        )

        point_dict = task.sample()
        results = task(point_dict)

        assert results[0]["type"] == "objective"
        first_objective = results[0]["value"]
        assert results[1]["type"] == "gradient"
        first_gradient = results[1]["value"]

        point_tuple = tuple(point_dict.values())
        second_point_array = np.array(point_tuple) - step_size * np.array(
            first_gradient
        )

        second_point = dict(zip(point_dict.keys(), second_point_array))

        if any(isinstance(dim, _Discrete) for dim in task.space.values()):
            # NOTE: Some dimensions don't quite work here because they are discrete: we can't really
            # use a point that has batch_size = 12.04 for example.
            assert second_point_array not in task.space
            second_point = {
                k: int(v)
                if isinstance(v, float) and isinstance(task.space[k], _Discrete)
                else v
                for k, v in second_point.items()
            }
            second_point_array = np.array(tuple(second_point.values()))

        assert second_point.values() in task.space
        assert second_point_array in task.space

        second_results = task(second_point)
        # Check that no warnings are raised if the point is in the space of the task.
        with pytest.warns(None) as record:
            second_results = task(second_point)
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
