""" Utilities used to test the different subclasses of the ProfetTask. """
from dataclasses import asdict, replace
from logging import getLogger as get_logger
from pathlib import Path
from typing import ClassVar, Type

import numpy as np
import pytest
from _pytest.tmpdir import TempPathFactory

from orion.algo.space import Space

logger = get_logger(__name__)

_devices = ["cpu"]
try:
    import torch

    if torch.cuda.is_available():
        _devices.append("cuda")
except ImportError:
    pytest.skip("skipping profet tests", allow_module_level=True)

from typing import List, Union, overload

from pytest_mock import MockerFixture

from orion.benchmark.task.profet.profet_task import MetaModelConfig, ProfetTask
from orion.core.worker.trial import Trial

from .conftest import REAL_PROFET_DATA_DIR, is_nonempty_dir


@overload
def sample(task: ProfetTask) -> Trial:
    ...


@overload
def sample(task: ProfetTask, n_samples: int) -> List[Trial]:
    ...


def sample(task: ProfetTask, n_samples: int = None) -> Union[Trial, List[Trial]]:
    """Draw random sample(s) from the space of this task.

    Samples a trial (dict of hyper-parameters) from the search space of this task.
    This dict can then be passed as an input to the task to get the objective:

    ```python
    x: Trial = sample(task)
    assert x in task.space
    y: float = task(**x.param)

    xs: List[Trial] = sample(task, n_samples=2)
    ys = [task(**x.param) for x in xs]
    ```

    NOTE: The randomness in the sampling is also handled correctly, by passing the local rng State
    of the task.

    Parameters
    ----------
    task
        A Profet task.
    n_samples
        The number of samples to be drawn. When None, a single trial is returned. When an integer
        is passed, a list of Trials is returned. Defaults to `None`.

    Returns
    -------
    trials
        Single `orion.core.worker.trial.Trial` when `n` is not passed, or list of `Trials`, when
        `n` is an integer. Each element is a separate sample of this space, a trial containing
        values associated with the corresponding dimension.
    """
    if n_samples is None:
        return task.space.sample(n_samples=1, seed=task._np_rng_state)[0]
    return task.space.sample(n_samples=n_samples, seed=task._np_rng_state)


class ProfetTaskTests:
    """Base class for testing Profet tasks."""

    Task: ClassVar[Type[ProfetTask]]

    @pytest.fixture()
    def profet_train_config(self):
        """Fixture that provides a configuration object for the Profet algorithm for testing."""
        quick_train_config = self.Task.ModelConfig(  # type: ignore
            task_id=0,
            seed=123,
            num_burnin_steps=10,
            max_samples=1000,
            max_iters=10,
            mcmc_thining=5,
            num_steps=100,
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
        fake_x, fake_y, fake_c = self.Task.ModelConfig().load_data(fake_input_dir)  # type: ignore
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
        """Test that the mock_load_data fixture returns real data when the input dir is non-empty,
        and that the means, dtypes, shapes, etc match between the real and fake data.
        """
        model_config = self.Task.ModelConfig()  # type: ignore

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

        min_y, max_y = model_config.y_min, model_config.y_max
        assert min_y <= real_y.min() and min_y <= fake_y.min()
        assert real_y.max() <= max_y and fake_y.max() <= max_y

        min_c, max_c = model_config.c_min, model_config.c_max
        assert min_c <= real_c.min() and min_c <= fake_c.min()
        assert real_c.max() <= max_c and fake_c.max() <= max_c

    @pytest.mark.filterwarnings("ignore:Checkpoint file")
    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("device_str", _devices)
    def test_attributes(
        self,
        profet_train_config: MetaModelConfig,
        checkpoint_dir: Path,
        profet_input_dir: Path,
        device_str: str,
        tmp_path_factory,
    ):
        """Simple test: Check that a task can be created and that its attributes are used and set
        correctly.
        """
        max_trials = 123
        device = torch.device(device_str)
        task = self.Task(
            max_trials=max_trials,
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        assert task.max_trials == max_trials
        assert task.input_dir == profet_input_dir
        assert task.checkpoint_dir == checkpoint_dir
        assert isinstance(task.device, torch.device)
        assert task.device == device
        assert task.device.type == device_str
        # NOTE: The class attributes might differ per task, but that's ok.
        assert isinstance(task.model_config, self.Task.ModelConfig)
        assert asdict(task.model_config) == asdict(profet_train_config)

    @pytest.mark.timeout(30)
    def test_instantiating_task(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        tmp_path_factory: TempPathFactory,
        checkpoint_dir: Path,
        mocker: MockerFixture,
    ):
        """Tests that when instantiating multiple tasks with the same arguments, the
        meta-model is trained only once.
        """
        # Directory should be empty.
        assert len(list(checkpoint_dir.iterdir())) == 0

        mocked_get_task_network = mocker.spy(profet_train_config, "get_task_network")
        mocked_load_task_network = mocker.spy(profet_train_config, "load_task_network")
        assert mocked_get_task_network.call_count == 0
        assert mocked_load_task_network.call_count == 0

        task_a = self.Task(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        assert mocked_get_task_network.call_count == 1
        mocked_get_task_network.assert_called_once_with(input_path=task_a.input_dir)
        assert mocked_load_task_network.call_count == 0

        # Directory should have one file (the trained model checkpoint).
        assert len(list(checkpoint_dir.iterdir())) == 1

        task_b = self.Task(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        assert task_a.configuration == task_b.configuration
        assert task_a.checkpoint_file == task_b.checkpoint_file

        assert mocked_get_task_network.call_count == 1
        assert mocked_load_task_network.call_count == 1
        mocked_load_task_network.assert_called_once_with(task_b.checkpoint_file)

        # Directory should *still* only have one file (the trained model checkpoint).
        assert len(list(checkpoint_dir.iterdir())) == 1

    @pytest.mark.parametrize("with_grad", [True, False])
    @pytest.mark.parametrize("device_str", _devices)
    def test_configuration(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        with_grad: bool,
        device_str: str,
    ):
        """Test that tasks have a proper configuration and that they can be created from it."""
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

        This is why I'm using the `sample` function above, so that I don't have to reach
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
        """Tests that two tasks with different arguments give different results."""
        first_task_kwargs = dict(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )

        first_task = self.Task(**first_task_kwargs)
        first_trial = sample(first_task)

        first_results = first_task(**first_trial.params)
        assert len(first_results) == 1
        assert first_results[0]["type"] == "objective"
        first_objective = first_results[0]["value"]

        second_task_model_config = replace(
            profet_train_config, seed=profet_train_config.seed + 456
        )
        second_task_kwargs = first_task_kwargs.copy()
        second_task_kwargs["model_config"] = second_task_model_config

        second_task = self.Task(**second_task_kwargs)
        second_trial = sample(second_task)
        assert second_trial != first_trial

        second_results = second_task(**second_trial.params)
        assert second_results != first_results

        assert len(second_results) == 1
        assert second_results[0]["type"] == "objective"
        second_objective = second_results[0]["value"]
        assert first_objective != second_objective

    @pytest.mark.parametrize("use_same_model", [True, False])
    @pytest.mark.parametrize("with_grad", [True, False])
    def test_call_is_reproducible(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        tmp_path: Path,
        use_same_model: bool,
        with_grad: bool,
    ):
        """Tests for the seeding of the model.
        When creating two tasks with the same parameters & seed:
        - When creating both models from scratch
        - When creating the first model from scratch, and loading the same model for the second task
        The model for both tasks should:
            - Have identical weights
            - give identical outputs for the same input.
        """
        # Create a new, empty training directory for this test.
        checkpoint_dir = tmp_path

        task_kwargs = dict(
            model_config=profet_train_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
            with_grad=with_grad,
        )
        task_a = self.Task(**task_kwargs)

        if use_same_model:
            # Use the same checkpoint directory, meaning that model for task #2 will be loaded from
            # the checkpoint created by task #1.
            task_b = self.Task(**task_kwargs)
        else:
            # NOTE: check if the training is seeded properly by using a different checkpoint
            # directory, forcing the task to create and train a new model using the same training
            # configuration, rather than load the same one as task_a.
            task_b_kwargs = task_kwargs.copy()
            task_b_kwargs["checkpoint_dir"] = checkpoint_dir.with_name(
                checkpoint_dir.name + "_2"
            )
            task_b = self.Task(**task_b_kwargs)

        # Check that, if the seed is the same, the weights are the same, regardless of if the model
        # is created from scratch or loaded from a checkpoint.
        task_a_params = dict(task_a.net.named_parameters())
        task_b_params = dict(task_b.net.named_parameters())
        assert task_a_params.keys() == task_b_params.keys()
        for param_name, task_a_param in task_a_params.items():
            task_b_param = task_b_params[param_name]
            assert torch.allclose(task_a_param, task_b_param), param_name
            if task_a_param.grad is not None:
                assert task_b_param.grad is not None
                assert torch.allclose(task_a_param.grad, task_b_param.grad), param_name

        trial_a = sample(task_a)
        trial_b = sample(task_b)
        assert trial_a.params == trial_b.params

        # NOTE: The forward pass samples from a distribution, therefore the values might be
        # different when calling the same task twice on the same input.
        # However, given two different task instances (with the same seed) they should give the same
        # exact sample.
        result_a = task_a(**trial_a.params)
        result_b = task_b(**trial_b.params)
        if not with_grad:
            # NOTE: Can't do this simple check with grads, bool of numpy arrays is ambiguous.
            assert result_a == result_b

        assert len(result_a) == 1 if not with_grad else 2
        assert result_a[0]["type"] == "objective"
        objective_a = result_a[0]["value"]
        objective_b = result_b[0]["value"]
        assert objective_a == objective_b

        if with_grad:
            assert result_a[1]["type"] == "gradient"
            grad_a = result_a[0]["value"]
            grad_b = result_b[0]["value"]
            assert np.allclose(grad_a, grad_b)

    @pytest.mark.parametrize("seed", [123, 456])
    def test_call_twice_with_same_task_gives_same_result(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        seed: int,
    ):
        """When using the same task and the same point twice, the results should be identical."""
        model_config = replace(profet_train_config, seed=seed)
        task = self.Task(
            model_config=model_config,
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
        )
        assert task.model_config.seed == seed
        trial = sample(task)

        first_results = task(**trial.params)
        assert len(first_results) == 1
        assert first_results[0]["type"] == "objective"

        # Pass the same point under different forms and check that the results are exactly the same.
        second_results = task(**trial.params)
        assert first_results == second_results

    @pytest.mark.parametrize("step_size", [1e-2, 1e-4])
    def test_call_with_gradients(
        self,
        profet_train_config: MetaModelConfig,
        profet_input_dir: Path,
        checkpoint_dir: Path,
        step_size: float,
    ):
        """Test that calling the task with the `with_grad` returns the gradient at that point."""
        task = self.Task(
            input_dir=profet_input_dir,
            checkpoint_dir=checkpoint_dir,
            model_config=profet_train_config,
            with_grad=True,
        )

        first_trial = sample(task)
        results = task(**first_trial.params)

        assert results[0]["type"] == "objective"
        first_objective = results[0]["value"]
        assert results[1]["type"] == "gradient"
        first_gradient = results[1]["value"]
        assert first_gradient is not None
        # Note: Gradients here might not have the same structure as the inputs, in the case of
        # XGBoost, since there are categorical variables.
