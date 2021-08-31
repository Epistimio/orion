""" Base class for Tasks that are generated using the Profet algorithm.

```
@misc{klein2019metasurrogate,
      title={Meta-Surrogate Benchmarking for Hyperparameter Optimization}, 
      author={Aaron Klein and Zhenwen Dai and Frank Hutter and Neil Lawrence and Javier Gonzalez},
      year={2019},
      eprint={1905.12982},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
"""
import functools
import hashlib
import json
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import GPy
import numpy as np
import torch
from GPy.models import BayesianGPLVM
from pybnn.bohamiann import Bohamiann
from torch import nn
from torch._C import is_anomaly_enabled
from torch.distributions import Normal

from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture
from emukit.examples.profet.meta_benchmarks.meta_forrester import get_architecture_forrester
from emukit.examples.profet.train_meta_model import download_data
from dataclasses import is_dataclass, asdict, dataclass

from logging import getLogger as get_logger
from orion.algo.space import Space
from orion.benchmark.task.base import BaseTask
from orion.benchmark.task.task_wrapper import params_to_array, trial_to_array
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.format_trials import dict_to_trial


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
    """ Configuration options for the training of the Profet meta-model. """

    n_samples: int = 1
    # TODO: Maybe could reduce this a bit to make the task generation faster?
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
    config: MetaModelTrainingConfig = None,
    hidden_space: int = 5,
    with_cost: bool = False,
    normalize_targets: bool = False,
) -> Tuple[Bohamiann, Optional[Bohamiann]]:
    config = config or MetaModelTrainingConfig()

    model_objective = Bohamiann(
        get_network=get_architecture, print_every_n_steps=1000, normalize_output=normalize_targets,
    )
    logger.info("Training Bohamiann objective model.")
    # TODO: With the FcNet task, the dataset has size 8_100_000, which takes a LOT of
    # memory to run!
    if config.max_samples is not None:
        logger.info(f"Limiting the dataset to a maximum of {config.max_samples} samples.")
        X_train = X_train[: config.max_samples, ...]
        Y_train = Y_train[: config.max_samples, ...]
    logger.info(X_train.shape, Y_train.shape, config, config.batch_size)
    model_objective.train(
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
        model_cost = Bohamiann(get_network=get_default_architecture, print_every_n_steps=1000)
        logger.info("Training Bohamiann cost model.")
        model_cost.train(
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
        model_cost = None

    return model_objective, model_cost


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


def get_task_network(
    input_path: Union[Path, str],
    benchmark: str,
    seed: int,
    task_id=None,
    config: MetaModelTrainingConfig = None,
) -> Tuple[nn.Module, np.ndarray]:
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
        hidden_space=hidden_space[benchmark],
        with_cost=False,
        config=config,
    )

    net = get_network(objective_model, X_train.shape[1])

    if task_id is None:
        task_id = rng.randint(Y.shape[0])

    multiplier = rng.randn(hidden_space[benchmark])
    h = task_features_mean[task_id] + task_features_std[task_id] * multiplier

    return net, h


def load_task_network(checkpoint_file: Union[str, Path]) -> Tuple[nn.Module, np.ndarray]:
    with open(checkpoint_file, "rb") as f:
        state = pickle.load(f)

    network = get_architecture[state["benchmark"]](state["size"])
    network.load_state_dict(state["network"])
    h = state["h"]

    return network, h


def compute_identity(size: int = 16, **sample) -> str:
    """Compute a unique hash out of a dictionary

    Parameters
    ----------
    size: int
        size of the unique hash

    **sample:
        Dictionary to compute the hash from

    """
    sample_hash = hashlib.sha256()

    for k, v in sorted(sample.items()):
        sample_hash.update(k.encode("utf8"))

        if isinstance(v, dict):
            sample_hash.update(compute_identity(size, **v).encode("utf8"))
        else:
            sample_hash.update(str(v).encode("utf8"))

    return sample_hash.hexdigest()[:size]


def save_task_network(
    checkpoint_file: Union[str, Path], benchmark: str, network: nn.Module, h: np.ndarray
) -> None:
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
        # TODO: Also save the train_config in the `configuration` of the task.
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

        checkpoint_file = self.checkpoint_dir / filename
        logger.info(f"Checkpoint file for this task: {checkpoint_file}")

        self.device: torch.device
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if os.path.exists(checkpoint_file):
            logger.info(f"Model has already been trained: loading it from file {checkpoint_file}.")
            self.net, self.h = load_task_network(checkpoint_file)
        else:
            logger.info(f"Checkpoint file {checkpoint_file} doesn't exist: re-training the model.")
            logger.info(f"Task hash params: {task_hash_params}")

            checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
            # Need to re-train the meta-model and sample this task.
            self.net, self.h = get_task_network(
                self.input_dir, benchmark, seed=seed, task_id=task_id, config=train_config
            )
            save_task_network(checkpoint_file, benchmark, self.net, self.h)

        self.net = self.net.to(device=self.device)
        self.h_tensor = torch.as_tensor(self.h, dtype=torch.float32, device=self.device)

        self._space = SpaceBuilder().build(self.get_search_space())

    def call(self, x: Dict, with_grad: bool = False) -> List[Dict]:
        if is_dataclass(x):
            x = asdict(x)

        logger.debug(f"received x={x}")
        # TODO: We don't use `self.space` here, because that attribute is
        # truncated whenever a value if fixed with `kwargs`.
        # NOTE: We assume here that keys are in order of insertion.
        x_np = params_to_array(x, space=self._space)

        x_tensor = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
        if with_grad:
            x_tensor.requires_grad_(True)

        p_tensor = torch.cat([x_tensor, self.h_tensor])[None, :]
        # p = np.concatenate((x_np, self.h))[None, :]

        # TODO: Bug when using forrester task:
        # RuntimeError: size mismatch, m1: [1 x 4], m2: [3 x 100]
        # p_tensor = torch.as_tensor(p, dtype=torch.float32)
        out = self.net(p_tensor)

        # TODO: Use a distribution, and use `rsample()`, so that we can also return the gradients if
        # need be.
        y_dist = Normal(loc=out[0, 0], scale=torch.exp(out[0, 1]))
        y_sample = y_dist.rsample()

        results: List[Dict] = []

        # TODO: Verify the validity of the results.
        # mean = out[0, 0].numpy()
        # var = np.exp(out[0, 1]).numpy()
        # y_sample = self.rng.normal(mean, var, size=1)
        name = (
            "profet"
            + "."
            + type(self).__name__.lower()
            + (f"_{self.task_id}" if self.task_id is not None else "")
        )
        results.append(dict(name=name, type="objective", value=float(y_sample)))

        if with_grad:
            results.append(dict(name=name, type="gradient", value=x_tensor.grad.cpu().numpy()))

        return results

    @property
    def configuration(self):
        """Return the configuration of the task."""
        return {
            self.__class__.__qualname__: self._param_names,
            "task_id": self.task_id,
            "input_dir": self.input_dir,
            "rng": self.rng,
            "seed": self.seed,
            "checkpoint_dir": self.checkpoint_dir,
            "max_trials": self.max_trials,
            "train_config": self.train_config,
        }
