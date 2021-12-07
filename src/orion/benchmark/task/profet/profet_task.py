""" Base class for Tasks that are generated using the Profet algorithm.

For more information on Profet, see original paper at https://arxiv.org/abs/1905.12982.

Klein, Aaron, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. "Meta-surrogate
benchmarking for hyperparameter optimization." Advances in Neural Information Processing Systems 32
(2019): 6270-6280.
"""
import os
import random
import warnings
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import asdict
from logging import getLogger as get_logger
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List, ClassVar, Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import torch
from torch.distributions import Normal

from orion.algo.space import Space
from orion.benchmark.task.base import BaseTask
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils import compute_identity
from orion.core.utils.format_trials import dict_to_trial, trial_to_tuple
from orion.core.utils.points import flatten_dims

from orion.benchmark.task.profet.model_utils import MetaModelConfig


logger = get_logger(__name__)


@contextmanager
def make_reproducible(seed: int):
    """ Makes the random operations within a block of code reproducible for a given seed. """
    # First: Get the starting random state, and restore it after.
    start_random_state = random.getstate()
    start_np_rng_state = np.random.get_state()
    with torch.random.fork_rng():
        # Set the random state, using the given seed.
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        yield

    # Restore the random state to the original state.
    np.random.set_state(start_np_rng_state)
    random.setstate(start_random_state)


InputType = TypeVar("InputType", bound=Dict[str, Any], covariant=True)


class ProfetTask(BaseTask, Generic[InputType]):
    """Base class for Tasks that are generated using the Profet algorithm.

    For more information on Profet, see original paper at https://arxiv.org/abs/1905.12982.

    Klein, Aaron, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. "Meta-surrogate benchmarking for 
    hyperparameter optimization." Advances in Neural Information Processing Systems 32 (2019): 6270-6280.
    
    Parameters
    ----------
    max_trials : int, optional
        Max number of trials to run, by default 100
    input_dir : Union[Path, str], optional
        Input directory containing the data used to train the meta-model, by default None.
    checkpoint_dir : Union[Path, str], optional
        Directory used to save/load trained meta-models, by default None.
    model_config : MetaModelConfig, optional
        Configuration options for the training of the meta-model, by default None
    device : Union[torch.device, str], optional
        The device to use for training, by default None.
    with_grad : bool, optional
        Wether the task should also return the gradients of the objective function with respect to
        the inputs. Defaults to `False`.
    """

    # Type of model config to use. Has to be overwritten by subclasses.
    ModelConfig: ClassVar[Type[MetaModelConfig]] = MetaModelConfig

    def __init__(
        self,
        max_trials: int = 100,
        input_dir: Union[Path, str] = "profet_data",
        checkpoint_dir: Union[Path, str] = None,
        model_config: MetaModelConfig = None,
        device: Union[torch.device, str] = None,
        with_grad: bool = False,
    ):
        super().__init__(max_trials=max_trials)
        self.input_dir = Path(input_dir)
        self.checkpoint_dir = Path(checkpoint_dir or self.input_dir / "checkpoints")

        # The config for the training of the meta-model.
        # NOTE: the train config is used to determine the hash of the task.
        if model_config is None:
            self.model_config = self.ModelConfig()
        elif isinstance(model_config, dict):
            # If passed a model config, for example through deserializing the configuration,
            # then convert it back to the right type, so the class attributes are correct.
            self.model_config = self.ModelConfig(**model_config)
        elif not isinstance(model_config, self.ModelConfig):
            self.model_config = self.ModelConfig(**asdict(model_config))
        else:
            self.model_config = model_config

        assert isinstance(self.model_config, self.ModelConfig)

        self.seed = self.model_config.seed

        self.with_grad = with_grad
        # The parameters that have an influence over the training of the meta-model are used to
        # create the filename where the model will be saved.
        task_hash_params = asdict(self.model_config)
        logger.info(f"Task hash params: {task_hash_params}")
        task_hash = compute_identity(**task_hash_params)

        filename = f"{task_hash}.pkl"

        self.checkpoint_file = self.checkpoint_dir / filename
        logger.info(f"Checkpoint file for this task: {self.checkpoint_file}")

        self.device: torch.device
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )

        # NOTE: Need to control the randomness that's happening inside *both* the training
        # function, as well as the loading function (since `load_task_network`` instantiates a model
        # and then loads the weights, it also affects the global rng state of pytorch).

        with make_reproducible(self.seed):
            if os.path.exists(self.checkpoint_file):
                logger.info(
                    f"Model has already been trained: loading it from file {self.checkpoint_file}."
                )
                self.net, h = self.model_config.load_task_network(self.checkpoint_file)
            else:
                warnings.warn(
                    RuntimeWarning(
                        f"Checkpoint file {self.checkpoint_file} doesn't exist: re-training the "
                        f"model. (This may take a *very* long time!)"
                    )
                )
                logger.info(f"Task hash params: {task_hash_params}")
                self.checkpoint_file.parent.mkdir(exist_ok=True, parents=True)

                # Need to re-train the meta-model and sample this task.
                self.net, h = self.model_config.get_task_network(self.input_dir)

        # Numpy random state. Currently only used in `sample()`
        self._np_rng_state = np.random.RandomState(self.seed)

        self.h: np.ndarray = np.array(h)
        self.model_config.save_task_network(self.checkpoint_file, self.net, self.h)

        self.net = self.net.to(device=self.device, dtype=torch.float32)
        self.net.eval()

        self.h_tensor = torch.as_tensor(self.h, dtype=torch.float32, device=self.device)

        self._space: Space = SpaceBuilder().build(self.get_search_space())
        self.name = f"profet.{type(self).__qualname__.lower()}_{self.model_config.task_id}"

    @property
    def space(self) -> Space:
        return self._space

    @overload
    def sample(self) -> InputType:
        ...

    @overload
    def sample(self, n: int) -> List[InputType]:
        ...

    def sample(self, n: int = None) -> Union[InputType, List[InputType]]:
        """ Samples a point (dict of hyper-parameters) from the search space of this task.

        This point can then be passed as an input to the task to get the objective.
        """
        if n is None:
            return self.sample(1)[0]
        return [
            dict(zip(self._space.keys(), point_tuple))  # type: ignore
            for point_tuple in self.space.sample(n, seed=self._np_rng_state)
        ]

    def call(self, x: InputType) -> List[Dict]:
        """Get the value of the sampled objective function at the given point (hyper-parameters).

        If `self.with_grad` is set, also returns the gradient of the objective function with respect
        to the inputs.

        Parameters
        ----------
        x : InputDict
            Dictionary of hyper-parameters.

        Returns
        -------
        List[Dict]
            Result dictionaries: objective and optionally gradient.

        Raises
        ------
        ValueError
            If the input isn't of a supported type.
        """
        logger.debug(f"received x: {x}")

        # A bit of gymnastics to convert the params Dict into a PyTorch tensor.
        trial = dict_to_trial(x, self._space)
        point_tuple = trial_to_tuple(trial, self._space)
        flattened_point = flatten_dims(point_tuple, self._space)
        x_tensor = torch.as_tensor(flattened_point).type_as(self.h_tensor)
        if self.with_grad:
            x_tensor = x_tensor.requires_grad_(True)
        p_tensor = torch.cat([x_tensor, self.h_tensor])
        p_tensor = torch.atleast_2d(p_tensor)

        devices = [] if self.device.type == "cpu" else [self.device]
        with torch.random.fork_rng(devices=devices):
            torch.random.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

            # Forward pass:
            out = self.net(p_tensor)

            # TODO: Double-check that the std is indeed in log form.
            y_mean, y_log_std = out[0, 0], out[0, 1]
            y_std = torch.exp(y_log_std)

            # NOTE: Here we create a distribution over `y`, and use `rsample()`, so that we get can also
            # return the gradients if need be.
            y_dist = Normal(loc=y_mean, scale=y_std)
            y_sample = y_dist.rsample()
            logger.debug(f"y_sample: {y_sample}")

        results: List[dict] = [
            dict(name=self.name, type="objective", value=y_sample.detach().cpu().item())
        ]

        if self.with_grad:
            self.net.zero_grad()
            y_sample.backward()
            results.append(
                dict(name=self.name, type="gradient", value=x_tensor.grad.cpu().numpy())
            )

        return results

    @property
    def configuration(self):
        """Return the configuration of the task."""
        return {
            self.__class__.__qualname__: {
                "max_trials": self.max_trials,
                "input_dir": str(self.input_dir),
                "checkpoint_dir": str(self.checkpoint_dir),
                "model_config": asdict(self.model_config),
                "device": self.device.type,
                "with_grad": self.with_grad,
            }
        }
