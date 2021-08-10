import functools
import hashlib
import json
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import GPy
import numpy as np
import torch
from GPy.models import BayesianGPLVM
from pybnn.bohamiann import Bohamiann
from torch import nn

from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture
from emukit.examples.profet.meta_benchmarks.meta_forrester import (
    get_architecture_forrester,
)
from emukit.examples.profet.train_meta_model import download_data
from dataclasses import is_dataclass, asdict, dataclass

# sys.path.extend([".", ".."])
from logging import getLogger as get_logger
from orion.benchmark.task.task import Task, compute_identity

logger = get_logger(__name__)


# TODO: Add support for forrester benchmark by adding the model_cost.
spaces: Dict[str, Dict[str, str]] = dict(
    forrester=dict(
        # TODO (@lebrice): This space is supposedly not correct. Need to look at
        # the profet paper in a bit more detail to check what the 'x' range is
        # supposed to be.
        # alpha='uniform(0, 1)',
        # beta='uniform(0, 1)'
        x="uniform(0, 1, discrete=False)",
    ),
    svm=dict(
        C="loguniform(np.exp(-10), np.exp(10))",
        gamma="loguniform(np.exp(-10), np.exp(10))",
    ),
    fcnet=dict(
        learning_rate="loguniform(1e-6, 1e-1)",
        batch_size="loguniform(8, 128, discrete=True)",
        units_layer1="loguniform(16, 512, discrete=True)",
        units_layer2="loguniform(16, 512, discrete=True)",
        dropout_rate_l1="uniform(0, 0.99)",
        dropout_rate_l2="uniform(0, 0.99)",
    ),
    xgboost=dict(
        learning_rate="loguniform(1e-6, 1e-1)",
        gamma="uniform(0, 2, discrete=False)",
        l1_regularization="loguniform(1e-5, 1e3)",
        l2_regularization="loguniform(1e-5, 1e3)",
        nb_estimators="uniform(10, 500, discrete=True)",
        subsampling="uniform(0.1, 1)",
        max_depth="uniform(1, 15, discrete=True)",
        min_child_weight="uniform(0, 20, discrete=True)",
    ),
)

get_architecture: Dict[str, Callable[[int], nn.Module]] = dict(
    forrester=get_architecture_forrester,
    svm=functools.partial(get_default_architecture, classification=True),
    fcnet=functools.partial(get_default_architecture, classification=True),
    xgboost=get_default_architecture,
)


names: Dict[str, str] = dict(
    forrester="data_sobol_forrester.json",
    svm="data_sobol_svm.json",
    fcnet="data_sobol_fcnet.json",
    xgboost="data_sobol_xgboost.json",
)

hidden_space: Dict[str, int] = dict(forrester=2, fcnet=5, svm=5, xgboost=5)

normalize_targets: Dict[str, bool] = dict(
    forrester=True, fcnet=False, svm=False, xgboost=True
)

log_cost: Dict[str, bool] = dict(forrester=False, fcnet=True, svm=True, xgboost=True)

log_target: Dict[str, bool] = dict(
    forrester=False, fcnet=False, svm=False, xgboost=True
)


def load_data(
    input_path: Union[str, Path], benchmark: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    file = Path(input_path) / "profet_data" / names[benchmark]
    if not file.exists():
        logger.info(f"File {file} doesn't exist, attempting to download data.")
        download_data(input_path)
        logger.info(f"Download finished.")
        assert file.exists(), input_path
    res = json.load(open(file, "r"))
    X, Y, C = np.array(res["X"]), np.array(res["Y"]), np.array(res["C"])
    if len(X.shape) == 1:
        X = X[:, None]

    return X, Y, C


def normalize_Y(
    Y: np.ndarray, indexD: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_idx = np.max(indexD)
    Y_mean = np.zeros(max_idx + 1)
    Y_std = np.zeros(max_idx + 1)
    for i in range(max_idx + 1):
        Y_mean[i] = Y[indexD == i].mean()
        Y_std[i] = Y[indexD == i].std() + 1e-8
        Y[indexD == i] = (Y[indexD == i] - Y_mean[i]) / Y_std[i]
    return Y, Y_mean[:, None], Y_std[:, None]


def get_features(
    X: np.ndarray, Y: np.ndarray, C: np.ndarray, hidden_space: int
) -> Tuple[np.ndarray, np.ndarray]:
    n_inducing_lvm = 50
    max_iters = 10000
    Q_h = hidden_space  # the dimensionality of the latent space
    n_tasks = Y.shape[0]
    n_configs = X.shape[0]
    index_task = np.repeat(np.arange(n_tasks), n_configs)
    Y_norm, _, _ = normalize_Y(deepcopy(Y.flatten()), index_task)

    # train the probabilistic encoder
    kern = GPy.kern.Matern52(Q_h, ARD=True)

    m_lvm = BayesianGPLVM(
        Y_norm.reshape(n_tasks, n_configs),
        Q_h,
        kernel=kern,
        num_inducing=n_inducing_lvm,
    )
    m_lvm.optimize(max_iters=max_iters, messages=1)

    ls = np.array(
        [m_lvm.kern.lengthscale[i] for i in range(m_lvm.kern.lengthscale.shape[0])]
    )

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tasks = Y.shape[0]
    n_samples_task = 500
    hidden_space = task_features_std.shape[1]
    n_configs = X.shape[0]

    X_train = []
    Y_train = []
    C_train = []

    for i, xi in enumerate(X):
        for idx in range(n_tasks):
            for _ in range(n_samples_task):
                ht = task_features_mean[idx] + task_features_std[idx] * np.random.randn(
                    hidden_space
                )

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


@dataclass
class MetaModelTrainingConfig:
    n_samples: int = 1
    # TODO: Maybe could reduce this a bit to make the task generation faster?
    num_burnin_steps: int = 50000
    num_steps: Optional[int] = None
    mcmc_thining: int = 100
    lr: float = 1e-2
    batch_size: int = 5
    # Maximum number of samples to use when training the meta-model. This can be useful
    # if the dataset is large (e.g. FCNet task) and you don't have crazy amounts of
    # memory. 
    max_samples: Optional[int] = None

    def __post_init__(self):
        if self.num_steps is None:
            self.num_steps = 100 * self.n_samples + 1
    


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
        get_network=get_architecture,
        print_every_n_steps=1000,
        normalize_output=normalize_targets,
    )
    print("Training Bohamiann objective model.")
    print(X_train.shape, Y_train.shape, config, config.batch_size)
    # TODO: With the FcNet task, the dataset has size 8_100_000, which takes a LOT of
    # memory to run!
    if config.max_samples is not None:
        print(f"Limiting the dataset to a maximum of {config.max_samples} samples.")
        X_train = X_train[:config.max_samples, ...]
        Y_train = Y_train[:config.max_samples, ...]
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
        model_cost = Bohamiann(
            get_network=get_default_architecture, print_every_n_steps=1000
        )
        print("Training Bohamiann cost model.")
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
    net = model.get_network(size)

    with torch.no_grad():
        weights = model.sampled_weights[idx]
        for parameter, sample in zip(net.parameters(), weights):
            parameter.copy_(torch.from_numpy(sample))
    return net


def get_task_network(
    input_path: Union[Path, str],
    benchmark: str,
    rng,
    task_id=None,
    config: MetaModelTrainingConfig = None,
) -> Tuple[nn.Module, np.ndarray]:
    config = config or MetaModelTrainingConfig()

    X, Y, C = load_data(input_path, benchmark)

    task_features_mean, task_features_std = get_features(
        X, Y, C, hidden_space[benchmark],
    )

    X_train, Y_train, C_train = get_training_data(
        X,
        Y,
        C,
        task_features_mean,
        task_features_std,
        log_C=log_cost[benchmark],
        log_Y=log_target[benchmark],
    )
    model_objective, model_cost = get_meta_model(
        X_train,
        Y_train,
        C_train,
        get_architecture[benchmark],
        normalize_targets=normalize_targets[benchmark],
        hidden_space=hidden_space[benchmark],
        with_cost=False,
        config=config,
    )

    net = get_network(model_objective, X_train.shape[1])

    if task_id is None:
        task_id = rng.randint(Y.shape[0])

    h = task_features_mean[task_id] + task_features_std[task_id] * rng.randn(
        hidden_space[benchmark]
    )

    return net, h


def load_task_network(
    checkpoint_file: Union[str, Path]
) -> Tuple[nn.Module, np.ndarray]:
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


class ProfetTask(Task):
    def __init__(
        self,
        benchmark: str,
        task_id: Optional[int]=0,
        input_dir: Union[Path, str] = None,
        rng: np.random.RandomState = None,
        seed: int = 1,
        checkpoint_dir: Union[Path, str] = None,
        max_trials: int = 100,
        train_config: MetaModelTrainingConfig = None,
        **kwargs,
    ):
        if rng is None:
            rng = np.random.RandomState(seed)
        task_id = task_id or 0
        super().__init__(
            task_id=task_id,
            rng=rng,
            seed=seed,
            max_trials=max_trials,
            fixed_dims=kwargs,
        )
        self.benchmark = benchmark
        if not input_dir:
            input_dir = Path(os.environ.get("DATA_DIR")) / "profet"
        else:
            input_dir = Path(input_dir)

        if checkpoint_dir is None:
            checkpoint_dir = input_dir / "checkpoints"
        # TODO: Save the train config in the `configuration` of the task.
        self.train_config = train_config or MetaModelTrainingConfig()
        task_hash_params = dict(
            benchmark=benchmark,
            task_id=task_id,
            rng=rng,
            seed=seed,
            **asdict(self.train_config),
        )
        task_hash = compute_identity(**task_hash_params)

        filename = f"{task_hash}.pkl"

        checkpoint_file = Path(checkpoint_dir) / filename
        logger.debug(f"Checkpoint file for this task: {checkpoint_file}")

        # The config for the training of the meta-model.
        # TODO: This should probably be a part of the hash for the "id" of the task!
        self.train_config = train_config

        if os.path.exists(checkpoint_file):
            self.net, self.h = load_task_network(checkpoint_file)
        else:
            logger.info(
                f"Checkpoint file {checkpoint_file} doesn't exist: re-training the meta-model."
            )
            # TODO: Figure out where/how to set the logger config.
            logger.info(f"Task hash params: {task_hash_params}")
            print(f"Task hash params:", json.dumps({k: str(v) for k, v in task_hash_params.items()}, indent="\t"))

            checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
            # Need to re-train the meta-model and sample this task.
            self.net, self.h = get_task_network(
                input_dir, benchmark, rng, task_id, config=train_config
            )
            save_task_network(checkpoint_file, benchmark, self.net, self.h)

    @property
    def hash(self) -> str:
        # TODO: Return a unique "hash"/id/key for this task

        if is_dataclass(self):
            return compute_identity(**asdict(self))
        return compute_identity(
            benchmark=self.benchmark,
            task_id=self.task_id,
            **self.fixed_values,
            seed=self.seed,
        )

    @property
    def space(self) -> Dict[str, str]:
        space = spaces[self.benchmark].copy()
        for k in self.fixed_values:
            if k in space:
                space.pop(k)
        return space

    @property
    def full_space(self) -> Dict[str, str]:
        return spaces[self.benchmark]

    def __call__(self, x: np.ndarray = None, **kwargs):
        if x is not None:
            assert not kwargs, "can't pass x and kwargs at the same time!"
            if isinstance(x, list):
                return [self(x_i, **kwargs) for x_i in x]
            hp_dict = self.to_hp_dict(x)
        else:
            hp_dict = kwargs

        logger.debug(
            f"received x={x}, kwargs: {kwargs}, fixed values: {self.fixed_values}"
        )
        hp_dict.update(self.fixed_values)

        # TODO: We don't use `self.space` here, because that attribute is
        # truncated whenever a value if fixed with `kwargs`.
        # assert set(kwargs.keys()) == set(self.full_space.keys()), f"{set(kwargs.keys())} != {set(spaces[self.benchmark].keys())}"
        # NOTE: We assume here that keys are in order of insertion.
        x = np.array([hp_dict[key] for key in self.full_space.keys()])
        p = np.concatenate((x, self.h))[None, :]

        with torch.no_grad():
            # BUG in pybnn/util/layers.py, where they have self.log_var be a
            # DoubleTensor while x is a FloatTensor! Here we overwrite the
            # forward method, instead of editing the code directly.
            from pybnn.util.layers import AppendLayer

            def forward(self, x):
                log_var = self.log_var.type_as(x)
                return torch.cat((x, log_var * torch.ones_like(x)), dim=1)

            AppendLayer.forward = forward

            # TODO: Bug when using forrester task:
            # RuntimeError: size mismatch, m1: [1 x 4], m2: [3 x 100]
            # debugging:
            # print(p.shape)
            # print(self.net)
            # print()
            out = self.net(torch.as_tensor(p, dtype=torch.float32))

        mean = out[0, 0].numpy()
        var = np.exp(out[0, 1]).numpy()

        # TODO: Verify the validity of the results.
        y_pred = self.rng.normal(mean, var, size=1)
        name = (
            "profet"
            + "."
            + type(self).__name__.lower()
            + (f"_{self.task_id}" if self.task_id is not None else "")
        )
        # TODO: Should we also return a 'gradient' results, since we could?
        return [dict(name=name, type="objective", value=float(y_pred))]


def main():
    data_path: Path = Path("profet_data/data")
    task = Task(data_path, benchmark="svm")
    import time

    start = time.time()
    print(task(C=0, gamma=0))
    print(task(C=0, gamma=0))
    print(task(C=0, gamma=0))
    print(task(C=0, gamma=0))
    print(task(C=0, gamma=0))
    print(task(C=0, gamma=0))
    print(task(C=0, gamma=0))
    print(task(C=1e-5, gamma=0))
    print(task(C=0, gamma=1e-5))
    print(task(C=1e-5, gamma=1e-5))
    print(time.time() - start)


if __name__ == "__main__":
    main()
