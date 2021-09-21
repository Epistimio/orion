""" Base class for Tasks that are generated using the Profet algorithm.

For more information on Profet, see original paper at https://arxiv.org/abs/1905.12982.

Klein, Aaron, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. "Meta-surrogate benchmarking for 
hyperparameter optimization." Advances in Neural Information Processing Systems 32 (2019): 6270-6280.
"""
import functools
import hashlib
import json
import os
import pickle
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import GPy
import numpy as np
import torch
from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture
from emukit.examples.profet.meta_benchmarks.meta_forrester import get_architecture_forrester
from emukit.examples.profet.train_meta_model import download_data
from GPy.models import BayesianGPLVM
from numpy.random import RandomState
from pybnn.bohamiann import Bohamiann
from torch import nn
from torch.distributions import Normal
from torch.functional import Tensor

from orion.algo.space import Space
from orion.benchmark.task.base import BaseTask
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.format_trials import dict_to_trial, trial_to_tuple
from orion.core.utils.points import flatten_dims
from orion.core.worker.trial import Trial
from orion.core.utils import compute_identity


logger = get_logger(__name__)


get_architecture: Dict[str, Callable[[int], nn.Module]] = dict(
    forrester=get_architecture_forrester,
    svm=functools.partial(get_default_architecture, classification=True),
    fcnet=functools.partial(get_default_architecture, classification=True),
    xgboost=get_default_architecture,
)


NAMES: Dict[str, str] = dict(
    forrester="data_sobol_forrester.json",
    svm="data_sobol_svm.json",
    fcnet="data_sobol_fcnet.json",
    xgboost="data_sobol_xgboost.json",
)

hidden_space: Dict[str, int] = dict(forrester=2, fcnet=5, svm=5, xgboost=5)

normalize_targets: Dict[str, bool] = dict(forrester=True, fcnet=False, svm=False, xgboost=True)

log_cost: Dict[str, bool] = dict(forrester=False, fcnet=True, svm=True, xgboost=True)

log_target: Dict[str, bool] = dict(forrester=False, fcnet=False, svm=False, xgboost=True)


@dataclass
class MetaModelTrainingConfig:
    """Configuration options for the training of the Profet meta-model."""

    n_samples: int = 1
    num_burnin_steps: int = 50000
    num_steps: int = 0
    mcmc_thining: int = 100
    lr: float = 1e-2
    batch_size: int = 5
    # Maximum number of samples to use when training the meta-model. This can be useful
    # if the dataset is large (e.g. FCNet task) and you don't have crazy amounts of
    # memory.
    max_samples: Optional[int] = None
    # Argument passed to the `BayesianGPLVM` constructor in `get_features`. Not sure what this does.
    n_inducing_lvm: int = 50
    # Argument passed to the `optimze` method of the `BayesianGPLVM` instance that is used in the
    # call to `get_features`. Appears to be the number of training iterations to perform.
    max_iters: int = 10_000
    # Number of tasks to create in `get_training_data`.
    n_samples_task: int = 500

    def __post_init__(self):
        if not self.num_steps:
            self.num_steps = 100 * self.n_samples + 1


def get_task_network(
    input_path: Union[Path, str],
    benchmark: str,
    seed: int,
    task_id: int,
    config: MetaModelTrainingConfig = None,
) -> Tuple[nn.Module, np.ndarray]:
    """Create, train and return a surrogate model for the given `benchmark`, `seed` and `task_id`.

    Parameters
    ----------
    input_path : Union[Path, str]
        Data directory containing the json files.
    benchmark : str
        Name of the benchmark.
    seed : int
        Seed for the random number generation.
    task_id : int
        Task index
    config : MetaModelTrainingConfig, optional
        Configuration options for the training of the meta-model, by default None, in which case the
        default options are used.

    Returns
    -------
    Tuple[nn.Module, np.ndarray]
        The surrogate model for the objective, as well as an array of sampled task features.
    """
    config = config or MetaModelTrainingConfig()
    rng = np.random.RandomState(seed=seed)

    X, Y, C = load_data(input_path, benchmark)

    task_features_mean, task_features_std = get_features(
        X=X,
        Y=Y,
        C=C,
        hidden_space=hidden_space[benchmark],
        n_inducing_lvm=config.n_inducing_lvm,
        max_iters=config.max_iters,
        display_messages=False,
    )

    X_train, Y_train, C_train = get_training_data(
        X,
        Y,
        C,
        task_features_mean,
        task_features_std,
        log_C=log_cost[benchmark],
        log_Y=log_target[benchmark],
        n_samples_task=config.n_samples_task,
    )
    objective_model, cost_model = get_meta_model(
        X_train,
        Y_train,
        C_train,
        get_architecture[benchmark],
        normalize_targets=normalize_targets[benchmark],
        with_cost=False,
        config=config,
    )

    net = get_network(objective_model, X_train.shape[1])

    multiplier = rng.randn(hidden_space[benchmark])
    h = task_features_mean[task_id] + task_features_std[task_id] * multiplier

    return net, h


def load_data(
    input_path: Union[str, Path], benchmark: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the profet data for the given benchmark from the input directory.

    When the input directory doesn't exist, attempts to download the data to create the input
    directory.

    Parameters
    ----------
    input_path : Union[str, Path]
        Input directory. Expects to find a json file for the given benchmark inside that directory.
    benchmark : str
        Name of the benchmark to load data for.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        X, Y, and C arrays.
    """
    file = Path(input_path) / NAMES[benchmark]
    if not file.exists():
        logger.info(f"File {file} doesn't exist, attempting to download data.")
        download_data(input_path)
        logger.info(f"Download finished.")
        if not file.exists():
            raise RuntimeError(f"Download finished, but file {file} still doesn't exist!")
    res = json.load(open(file, "r"))
    X, Y, C = np.array(res["X"]), np.array(res["Y"]), np.array(res["C"])
    if len(X.shape) == 1:
        X = X[:, None]

    return X, Y, C


def normalize_Y(Y: np.ndarray, indexD: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize the Y array and return its mean and standard deviations.

    Parameters
    ----------
    Y : np.ndarray
        Labels from the datasets.
    indexD : np.ndarray
        NOTE: Not sure what this argument represents.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing the Y array, the mean array, and the std array.
    """
    max_idx = np.max(indexD)
    Y_mean = np.zeros(max_idx + 1)
    Y_std = np.zeros(max_idx + 1)
    for i in range(max_idx + 1):
        Y_mean[i] = Y[indexD == i].mean()
        Y_std[i] = Y[indexD == i].std() + 1e-8
        Y[indexD == i] = (Y[indexD == i] - Y_mean[i]) / Y_std[i]
    return Y, Y_mean[:, None], Y_std[:, None]


def get_features(
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    hidden_space: int,
    n_inducing_lvm: int = 50,
    max_iters: int = 10000,
    display_messages: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate features for the given task.

    Parameters
    ----------
    X : np.ndarray
        Training examples
    Y : np.ndarray
        Training labels
    C : np.ndarray
        Training costs (NOTE: This isn't really used at the moment).
    hidden_space : int
        Dimensionality of the hidden space.
    n_inducing_lvm : int, optional
        Argument to the `BayesianGPLVM` constructor, by default 50.
    max_iters : int, optional
        Argument to `optimize` method of `BayesianGPLVM` that is called in `get_features`.
        Defaults to 10000.
    display_messages : bool, optional
        Wether to log messages to the console or not, by default True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The features mean and std arrays.
    """
    Q_h = hidden_space  # the dimensionality of the latent space
    n_tasks = Y.shape[0]
    n_configs = X.shape[0]
    index_task = np.repeat(np.arange(n_tasks), n_configs)
    Y_norm, _, _ = normalize_Y(deepcopy(Y.flatten()), index_task)

    # train the probabilistic encoder
    kern = GPy.kern.Matern52(Q_h, ARD=True)

    m_lvm = BayesianGPLVM(
        Y_norm.reshape(n_tasks, n_configs), Q_h, kernel=kern, num_inducing=n_inducing_lvm,
    )
    m_lvm.optimize(max_iters=max_iters, messages=display_messages)

    ls = np.array([m_lvm.kern.lengthscale[i] for i in range(m_lvm.kern.lengthscale.shape[0])])

    # generate data to train the multi-task model
    task_features_mean = np.array(m_lvm.X.mean / ls)
    task_features_std = np.array(np.sqrt(m_lvm.X.variance) / ls)

    return task_features_mean, task_features_std


def get_training_data(
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    task_features_mean: np.ndarray,
    task_features_std: np.ndarray,
    log_C: bool = True,
    log_Y: bool = False,
    n_samples_task: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create training data by sampling a given number of tasks.

    Parameters
    ----------
    X : np.ndarray
        Training examples
    Y : np.ndarray
        Training labels
    C : np.ndarray
        Training costs (NOTE: This isn't really used at the moment).
    task_features_mean : np.ndarray
        Mean of the model training weights.
    task_features_std : np.ndarray
        Std of the model training weights.
    log_C : bool, optional
        Wether to apply `np.log` onto `C`, by default True.
    log_Y : bool, optional
        Wether to apply `np.log` onto `Y`, by default False.
    n_samples_task : int, optional
        Number of tasks to sample, by default 500.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        numpy arrays containing the X, Y, and C's for each task.
    """
    n_tasks = Y.shape[0]
    hidden_space = task_features_std.shape[1]
    n_configs = X.shape[0]

    X_train = []
    Y_train = []
    C_train = []

    for i, xi in enumerate(X):
        for idx in range(n_tasks):
            for _ in range(n_samples_task):
                multiplier = np.random.randn(hidden_space)
                ht = task_features_mean[idx] + task_features_std[idx] * multiplier

                x = np.concatenate((xi, ht), axis=0)
                X_train.append(x)
                Y_train.append(Y[idx, i])
                C_train.append(C[idx, i])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    C_train = np.array(C_train)
    if log_C:
        C_train = np.log(C_train)

    if log_Y:
        Y_train = np.log(Y_train)

    return X_train, Y_train, C_train


def get_meta_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    C_train: np.ndarray,
    get_architecture: Callable,
    config: MetaModelTrainingConfig,
    with_cost: bool = False,
    normalize_targets: bool = False,
) -> Tuple[Bohamiann, Optional[Bohamiann]]:
    """Create, train and return the objective model, and (optionally) a cost model for the data.

    Parameters
    ----------
    X_train : np.ndarray
        Training samples.
    Y_train : np.ndarray
        Training objectives.
    C_train : np.ndarray
        Training costs.
    get_architecture : Callable
        Function used to get a model architecture for a given input dimensionality.
    config : MetaModelTrainingConfig
        Configuration options for the training of the meta-model.
    with_cost : bool, optional
        Wether to also create a surrogate model for the cost. Defaults to `False`.
    normalize_targets : bool, optional
        Wether to normalize the targets (y), by default False.

    Returns
    -------
    Tuple[Bohamiann, Optional[Bohamiann]]
        Surrogate model for the objective, as well as another for the cost, if `with_cose` is True,
        otherwise `None`.
    """

    objective_model = Bohamiann(
        get_network=get_architecture, print_every_n_steps=1000, normalize_output=normalize_targets,
    )
    logger.info("Training Bohamiann objective model.")
    # NOTE: With the FcNet task, the dataset has size 8_100_000, which takes a LOT of
    # memory to run!
    if config.max_samples is not None:
        logger.info(f"Limiting the dataset to a maximum of {config.max_samples} samples.")
        X_train = X_train[: config.max_samples, ...]
        Y_train = Y_train[: config.max_samples, ...]
        C_train = C_train[: config.max_samples, ...]

    logger.debug(f"Shapes: {X_train.shape}, {Y_train.shape}")
    logger.debug(f"config: {config}")

    objective_model.train(
        X_train,
        Y_train,
        num_steps=config.num_steps + config.num_burnin_steps,
        num_burn_in_steps=config.num_burnin_steps,
        keep_every=config.mcmc_thining,
        lr=config.lr,
        verbose=True,
        batch_size=config.batch_size,
    )

    if with_cost:
        cost_model = Bohamiann(get_network=get_default_architecture, print_every_n_steps=1000)
        logger.info("Training Bohamiann cost model.")
        cost_model.train(
            X_train,
            C_train,
            num_steps=config.num_steps + config.num_burnin_steps,
            num_burn_in_steps=config.num_burnin_steps,
            keep_every=config.mcmc_thining,
            lr=config.lr,
            verbose=True,
            batch_size=config.batch_size,
        )
    else:
        cost_model = None

    return objective_model, cost_model


def get_network(model: Bohamiann, size: int, idx: int = 0) -> nn.Module:
    """Retrieve a network with sampled weights for the given task id.

    Parameters
    ----------
    model : Bohamiann
        "Base" Bohamiann model used to get a network and its weights.
    size : int
        Input dimensions for the generated network.
    idx : int, optional
        Task idx, by default 0

    Returns
    -------
    nn.Module
        A `nn.Module` with sampled weights.
    """
    net = model.get_network(size)

    with torch.no_grad():
        weights = model.sampled_weights[idx]
        for parameter, sample in zip(net.parameters(), weights):
            parameter.copy_(torch.from_numpy(sample))
    return net


def load_task_network(checkpoint_file: Union[str, Path]) -> Tuple[nn.Module, np.ndarray]:
    """Load the result of the `get_task_network` function stored in the pickle file.

    Parameters
    ----------
    checkpoint_file : Union[str, Path]
        Path to a pickle file. The file is expected to contain a serialized dictionary, with keys
        "benchmark", "size", "network", and "h".

    Returns
    -------
    Tuple[nn.Module, np.ndarray]
        The surrogate model for the objective, as well as an array of sampled task features.
    """
    with open(checkpoint_file, "rb") as f:
        state = pickle.load(f)

    network = get_architecture[state["benchmark"]](state["size"])
    network.load_state_dict(state["network"])
    h = state["h"]

    return network, h


def save_task_network(
    checkpoint_file: Union[str, Path], benchmark: str, network: nn.Module, h: np.ndarray
) -> None:
    """Save the meta-model for the task at the given path. 

    Parameters
    ----------
    checkpoint_file : Union[str, Path]
        Path where the model should be saved
    benchmark : str
        The name of the benchmark
    network : nn.Module
        The network
    h : np.ndarray
        The embedding vector
    """
    checkpoint_file = Path(checkpoint_file)
    state = dict(
        benchmark=benchmark,
        network=network.state_dict(),
        size=list(network.parameters())[0].size()[1],
        h=h.tolist(),
    )

    tmp_file = checkpoint_file.with_suffix(".tmp")
    with open(tmp_file, "wb") as file:
        pickle.dump(state, file, protocol=pickle.DEFAULT_PROTOCOL)
    tmp_file.rename(checkpoint_file)


# BUG in pybnn/util/layers.py, where they have self.log_var be a
# DoubleTensor while x is a FloatTensor! Here we overwrite the
# forward method, instead of editing the code directly.
from pybnn.util.layers import AppendLayer


@functools.wraps(AppendLayer.forward)
def _patched_forward(self, x):
    log_var = self.log_var.type_as(x)
    return torch.cat((x, log_var * torch.ones_like(x)), dim=1)


AppendLayer.forward = _patched_forward


InputType = TypeVar("InputType")


class ProfetTask(BaseTask, Generic[InputType]):
    """Base class for Tasks that are generated using the Profet algorithm.

    For more information on Profet, see original paper at https://arxiv.org/abs/1905.12982.

    Klein, Aaron, Zhenwen Dai, Frank Hutter, Neil Lawrence, and Javier Gonzalez. "Meta-surrogate benchmarking for 
    hyperparameter optimization." Advances in Neural Information Processing Systems 32 (2019): 6270-6280.
    
    Parameters
    ----------
    benchmark : str
        Name of the benchmark.
    max_trials : int, optional
        Max number of trials to run, by default 100
    task_id : int, optional
        Task index, by default 0
    seed : int, optional
        Random seed, by default 123
    input_dir : Union[Path, str], optional
        Input directory containing the data used to train the meta-model, by default None.
    checkpoint_dir : Union[Path, str], optional
        Directory used to save/load trained meta-models, by default None.
    train_config : MetaModelTrainingConfig, optional
        Configuration options for the training of the meta-model, by default None
    device : Union[torch.device, str], optional
        The device to use for training, by default None.
    """

    def __init__(
        self,
        benchmark: str,
        max_trials: int = 100,
        task_id: int = 0,
        seed: int = 123,
        input_dir: Union[Path, str] = None,
        checkpoint_dir: Union[Path, str] = None,
        train_config: MetaModelTrainingConfig = None,
        device: Union[torch.device, str] = None,
    ):
        super().__init__(max_trials=max_trials)
        self.benchmark = benchmark
        self.task_id = task_id
        self.seed = seed
        self.input_dir = Path(input_dir if input_dir else "profet_data")
        self.checkpoint_dir = Path(checkpoint_dir or self.input_dir / "checkpoints")
        # The config for the training of the meta-model.
        # NOTE: the train config is used to determine the hash of the task.
        self.train_config = train_config or MetaModelTrainingConfig()

        # The parameters that have an influence over the training of the meta-model are used to
        # create the filename where the model will be saved.
        task_hash_params = dict(
            benchmark=self.benchmark,
            task_id=self.task_id,
            seed=self.seed,
            **asdict(self.train_config),
        )
        logger.info(f"Task hash params: {task_hash_params}")
        task_hash = compute_identity(**task_hash_params)

        filename = f"{task_hash}.pkl"

        self.checkpoint_file = self.checkpoint_dir / filename
        logger.info(f"Checkpoint file for this task: {self.checkpoint_file}")

        self.device: torch.device
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # TODO: Save those in the configuration dict?
        self._np_rng: Optional[np.random.RandomState] = None
        self._torch_rng_state: Optional[Tensor] = None
        with self.seed_randomness():
            if os.path.exists(self.checkpoint_file):
                logger.info(
                    f"Model has already been trained: loading it from file {self.checkpoint_file}."
                )
                self.net, self.h = load_task_network(self.checkpoint_file)
            else:
                logger.info(
                    f"Checkpoint file {self.checkpoint_file} doesn't exist: re-training the model."
                )
                logger.info(f"Task hash params: {task_hash_params}")

                self.checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
                # Need to re-train the meta-model and sample this task.
                self.net, self.h = get_task_network(
                    self.input_dir,
                    self.benchmark,
                    seed=self.seed,
                    task_id=self.task_id,
                    config=self.train_config,
                )
                save_task_network(self.checkpoint_file, self.benchmark, self.net, self.h)

        self.net = self.net.to(device=self.device)
        self.net.eval()
        self.h_tensor = torch.as_tensor(self.h, dtype=torch.float32, device=self.device)

        self._space: Space = SpaceBuilder().build(self.get_search_space())
        # IDEA: Could 'hack' the `sample` function of this Space instance so it always uses our
        # numpy rng state.
        # _original_sample = self._space.sample

        # def seeded_sample(_, n_samples: int = 1, seed=None):
        #     nonlocal self
        #     if seed is None:
        #         seed = self._np_rng
        #     return _original_sample(n_samples=n_samples, seed=seed)

        # self._space.sample = seeded_sample

        self.name = f"profet.{type(self).__qualname__.lower()}_{self.task_id}"

    @contextmanager
    def seed_randomness(self):
        """Context manager used to seed the portions of the code that use randomness, without
        affecting the global state.
        """
        devices = [] if self.device.type == "cpu" else [self.device]
        if self._np_rng is None:
            # Initialize the numpy random state.
            self._np_rng = RandomState(seed=self.seed)

        start_np_state = np.random.get_state()
        # Temporarily set the global random state to that of our RNG.
        np.random.set_state(self._np_rng.get_state())

        # Need to do this when it comes to the torch random state, since `torch.distributions` uses
        # the global state when sampling, and there are no ways of passing a seed to a distribution.
        with torch.random.fork_rng(devices=devices):
            if self._torch_rng_state is None:
                torch.random.manual_seed(self.seed)
            else:
                torch.random.set_rng_state(self._torch_rng_state)

            # Yield, which probably executes a few lines with randomness.
            yield

            # Update the torch random state.
            self._torch_rng_state = torch.random.get_rng_state()

        # NOTE: Re-creating a RandomState object and setting the state, since I don't know how to
        # get the 'global' RandomState object.
        self._np_rng = RandomState(seed=self.seed)
        self._np_rng.set_state(np.random.get_state())

        np.random.set_state(start_np_state)

    def call(
        self, x: Union[InputType, Trial, Dict, Tuple] = None, with_grad: bool = False, **kwargs,
    ) -> List[Dict]:
        """Get the value of the sampled objective function at the given point (hyper-parameters).

        If `with_grad` is passed, also returns the gradient of the objective function with respect
        to the inputs.

        Parameters
        ----------
        x : Union[InputType, Trial, Dict, Tuple]
            Either a Trial, a dataclass, a dict, or a tuple, that is a point from this tasks' space.
        with_grad : bool, optional
            Wether to also compute the gradients with respect to the inputs, by default False.

        Returns
        -------
        List[Dict]
            Result dictionaries: objective and optionally gradient.

        Raises
        ------
        ValueError
            If the input isn't of a supported type.
        """
        logger.debug(f"received x={x}")
        if x is None and kwargs:
            x = kwargs

        if is_dataclass(x):
            x = asdict(x)
        elif isinstance(x, Trial):
            x = x.params
        elif isinstance(x, tuple) and len(x) == len(self._space):
            x = dict(zip(self._space.keys(), x))
        elif isinstance(x, dict):
            # NOTE: Passing ndarrays and tensors is useful for testing, but this task now probably
            # accepts too many types of inputs.
            x = x
        elif isinstance(x, (np.ndarray, Tensor)):
            if x not in self._space:
                warnings.warn(
                    RuntimeWarning(
                        f"Point {x} isn't a valid point of the space {self._space}, but will still "
                        f"be passed to the task's model."
                    )
                )
        else:
            raise ValueError(
                f"Expected `x` to be a dataclass, a Trial, a dict, or a tuple of length "
                f"{len(self._space)}, but got {x} instead."
            )

        if isinstance(x, dict):
            # A bit of gymnastics to convert the params Dict into a numpy array.
            trial = dict_to_trial(x, self._space)
            point_tuple = trial_to_tuple(trial, self._space)
            flattened_point = flatten_dims(point_tuple, self._space)
            x = np.array(flattened_point)

        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        assert isinstance(x, Tensor)  # Just to keep the type checker happy: this is always true.
        x_tensor: Tensor = x
        if with_grad:
            x_tensor.requires_grad_(True)

        x_tensor = x_tensor.type_as(self.h_tensor)
        p_tensor = torch.cat([x_tensor, self.h_tensor])
        p_tensor = torch.atleast_2d(p_tensor)

        with self.seed_randomness():
            # TODO: Bug when using forrester task:
            # RuntimeError: size mismatch, m1: [1 x 4], m2: [3 x 100]
            out = self.net(p_tensor)
            # TODO: Double-check that the std is indeed in log form.
            y_mean, y_log_std = out[0, 0], out[0, 1]
            y_std = torch.exp(y_log_std)

            # NOTE: Here we create a distribution over `y`, and use `rsample()`, so that we get can also
            # return the gradients if need be.
            y_dist = Normal(loc=y_mean, scale=y_std)

            y_sample = y_dist.rsample()
            logger.debug(f"y_sample: {y_sample}")

        results: List[Dict] = []

        results.append(dict(name=self.name, type="objective", value=float(y_sample)))

        if with_grad:
            self.net.zero_grad()
            y_sample.backward()
            results.append(dict(name=self.name, type="gradient", value=x_tensor.grad.cpu().numpy()))

        return results

    @property
    def configuration(self):
        """Return the configuration of the task."""
        # TODO: Still not sure I understand how this mess works.
        return {
            self.__class__.__qualname__: {
                "max_trials": self.max_trials,
                "task_id": self.task_id,
                "seed": self.seed,
                "input_dir": self.input_dir,
                "checkpoint_dir": self.checkpoint_dir,
                "train_config": self.train_config,
            }
        }
