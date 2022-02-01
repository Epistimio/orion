import json
import pickle
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Optional,
    Tuple,
    Union,
)

import GPy
import numpy as np
import torch
from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture
from emukit.examples.profet.train_meta_model import download_data
from GPy.models import BayesianGPLVM
from pybnn.bohamiann import Bohamiann
from torch import nn


logger = get_logger(__name__)

# NOTE: Split up these shared properties into the different config classes.
# NAMES: Dict[str, str] = dict(
#     forrester="data_sobol_forrester.json",
#     svm="data_sobol_svm.json",
#     fcnet="data_sobol_fcnet.json",
#     xgboost="data_sobol_xgboost.json",
# )
# get_architecture: Dict[str, Callable[[int], nn.Module]] = dict(
#     forrester=get_architecture_forrester,
#     svm=functools.partial(get_default_architecture, classification=True),
#     fcnet=functools.partial(get_default_architecture, classification=True),
#     xgboost=get_default_architecture,
# )
# hidden_space: Dict[str, int] = dict(forrester=2, fcnet=5, svm=5, xgboost=5)
# normalize_targets: Dict[str, bool] = dict(forrester=True, fcnet=False, svm=False, xgboost=True)
# log_cost: Dict[str, bool] = dict(forrester=False, fcnet=True, svm=True, xgboost=True)
# log_target: Dict[str, bool] = dict(forrester=False, fcnet=False, svm=False, xgboost=True)

from abc import ABC


@dataclass
class MetaModelConfig(ABC):
    """Configuration options for the training of the Profet meta-model."""

    benchmark: str

    # ---------- "Abstract"/required attributes:
    json_file_name: ClassVar[str]
    get_architecture: ClassVar[Callable[[int], nn.Module]]
    hidden_space: ClassVar[int]
    log_cost: ClassVar[bool]
    log_target: ClassVar[bool]
    normalize_targets: ClassVar[bool]
    # -----------

    task_id: int = 0
    seed: int = 123

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
        self,
        input_path: Union[Path, str],
    ) -> Tuple[nn.Module, np.ndarray]:
        """Create, train and return a surrogate model for the given `benchmark`, `seed` and `task_id`.

        Parameters
        ----------
        input_path : Union[Path, str]
            Data directory containing the json files.

        Returns
        -------
        Tuple[nn.Module, np.ndarray]
            The surrogate model for the objective, as well as an array of sampled task features.
        """
        rng = np.random.RandomState(seed=self.seed)

        X, Y, C = self.load_data(input_path)

        task_features_mean, task_features_std = self.get_features(
            X=X,
            Y=Y,
            C=C,
            hidden_space=self.hidden_space,
            n_inducing_lvm=self.n_inducing_lvm,
            max_iters=self.max_iters,
            display_messages=False,
        )

        X_train, Y_train, C_train = self.get_training_data(
            X,
            Y,
            C,
            task_features_mean=task_features_mean,
            task_features_std=task_features_std,
            log_C=self.log_cost,
            log_Y=self.log_target,
            n_samples_task=self.n_samples_task,
        )
        objective_model, cost_model = self.get_meta_model(
            X_train,
            Y_train,
            C_train,
            with_cost=False,
        )

        net = self.get_network(objective_model, X_train.shape[1])

        multiplier = rng.randn(self.hidden_space)
        h = task_features_mean[self.task_id] + task_features_std[self.task_id] * multiplier
        return net, h

    def load_data(self, input_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the profet data for the given benchmark from the input directory.

        When the input directory doesn't exist, attempts to download the data to create the input
        directory.

        Parameters
        ----------
        input_path : Union[str, Path]
            Input directory. Expects to find a json file for the given benchmark inside that directory.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            X, Y, and C arrays.
        """
        # file = Path(input_path) / NAMES[benchmark]
        file = Path(input_path) / self.json_file_name
        if not file.exists():
            logger.info(f"File {file} doesn't exist, attempting to download data.")
            download_data(input_path)
            logger.info("Download finished.")
            if not file.exists():
                raise RuntimeError(f"Download finished, but file {file} still doesn't exist!")
        with open(file, "r") as f:
            res = json.load(f)
        X, Y, C = np.array(res["X"]), np.array(res["Y"]), np.array(res["C"])
        if len(X.shape) == 1:
            X = X[:, None]
        return X, Y, C

    def normalize_Y(
        self, Y: np.ndarray, indexD: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        self,
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
            Whether to log messages to the console or not, by default True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The features mean and std arrays.
        """
        Q_h = hidden_space  # the dimensionality of the latent space
        n_tasks = Y.shape[0]
        n_configs = X.shape[0]
        index_task = np.repeat(np.arange(n_tasks), n_configs)
        Y_norm, _, _ = self.normalize_Y(deepcopy(Y.flatten()), index_task)

        # train the probabilistic encoder
        kern = GPy.kern.Matern52(Q_h, ARD=True)

        m_lvm = BayesianGPLVM(
            Y_norm.reshape(n_tasks, n_configs),
            Q_h,
            kernel=kern,
            num_inducing=n_inducing_lvm,
        )
        m_lvm.optimize(max_iters=max_iters, messages=display_messages)

        ls = np.array([m_lvm.kern.lengthscale[i] for i in range(m_lvm.kern.lengthscale.shape[0])])

        # generate data to train the multi-task model
        task_features_mean = np.array(m_lvm.X.mean / ls)
        task_features_std = np.array(np.sqrt(m_lvm.X.variance) / ls)

        return task_features_mean, task_features_std

    def get_training_data(
        self,
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
            Whether to apply `np.log` onto `C`, by default True.
        log_Y : bool, optional
            Whether to apply `np.log` onto `Y`, by default False.
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
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        C_train: np.ndarray,
        with_cost: bool = False,
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
            Whether to also create a surrogate model for the cost. Defaults to `False`.
        normalize_targets : bool, optional
            Whether to normalize the targets (y), by default False.

        Returns
        -------
        Tuple[Bohamiann, Optional[Bohamiann]]
            Surrogate model for the objective, as well as another for the cost, if `with_cose` is True,
            otherwise `None`.
        """
        objective_model = Bohamiann(
            get_network=type(self).get_architecture,
            print_every_n_steps=1000,
            normalize_output=self.normalize_targets,
        )
        logger.info("Training Bohamiann objective model.")
        # NOTE: With the FcNet task, the dataset has size 8_100_000, which takes a LOT of
        # memory to run!
        if self.max_samples is not None:
            logger.info(f"Limiting the dataset to a maximum of {self.max_samples} samples.")
            X_train = X_train[: self.max_samples, ...]
            Y_train = Y_train[: self.max_samples, ...]
            C_train = C_train[: self.max_samples, ...]

        logger.debug(f"Shapes: {X_train.shape}, {Y_train.shape}")
        logger.debug(f"config: {self}")

        objective_model.train(
            X_train,
            Y_train,
            num_steps=self.num_steps + self.num_burnin_steps,
            num_burn_in_steps=self.num_burnin_steps,
            keep_every=self.mcmc_thining,
            lr=self.lr,
            verbose=True,
            batch_size=self.batch_size,
        )

        if with_cost:
            cost_model = Bohamiann(get_network=get_default_architecture, print_every_n_steps=1000)
            logger.info("Training Bohamiann cost model.")
            cost_model.train(
                X_train,
                C_train,
                num_steps=self.num_steps + self.num_burnin_steps,
                num_burn_in_steps=self.num_burnin_steps,
                keep_every=self.mcmc_thining,
                lr=self.lr,
                verbose=True,
                batch_size=self.batch_size,
            )
        else:
            cost_model = None

        return objective_model, cost_model

    def get_network(self, model: Bohamiann, size: int, idx: int = 0) -> nn.Module:
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

    def load_task_network(
        self,
        checkpoint_file: Union[str, Path],
    ) -> Tuple[nn.Module, np.ndarray]:
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

        if state["benchmark"] != self.benchmark:
            raise RuntimeError(
                f"Trying to load model for benchmark {self.benchmark} from checkpoint that "
                f"contains data from benchmark {state['benchmark']}."
            )
        network = type(self).get_architecture(state["size"])
        network.load_state_dict(state["network"])
        h = state["h"]

        return network, h

    def save_task_network(
        self, checkpoint_file: Union[str, Path], network: nn.Module, h: np.ndarray
    ) -> None:
        """Save the meta-model for the task at the given path.

        Parameters
        ----------
        checkpoint_file : Union[str, Path]
            Path where the model should be saved
        network : nn.Module
            The network
        h : np.ndarray
            The embedding vector
        """
        checkpoint_file = Path(checkpoint_file)
        state = dict(
            benchmark=self.benchmark,
            network=network.state_dict(),
            size=list(network.parameters())[0].size()[1],
            h=h.tolist(),
        )

        tmp_file = checkpoint_file.with_suffix(".tmp")
        with open(tmp_file, "wb") as file:
            pickle.dump(state, file, protocol=pickle.DEFAULT_PROTOCOL)
        tmp_file.rename(checkpoint_file)
