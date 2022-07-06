""" Options and utilities for training the profet meta-model from Emukit. """
import json
import pickle
import warnings
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional, Tuple, Union

import numpy as np

_ERROR_MSG = (
    "The `profet` extras needs to be installed in order to use the Profet tasks.\n"
    "Error: {0}\n"
    "Use `pip install orion[profet]` to install the profet extras."
)
try:
    import GPy
    import torch
    from emukit.examples.profet.meta_benchmarks.architecture import (
        get_default_architecture,
    )
    from emukit.examples.profet.meta_benchmarks.meta_forrester import (
        get_architecture_forrester,  # type: ignore
    )
    from emukit.examples.profet.train_meta_model import download_data
    from GPy.models import BayesianGPLVM
    from pybnn.bohamiann import Bohamiann
except ImportError as err:
    warnings.warn(RuntimeWarning(_ERROR_MSG.format(err)))
    # NOTE: Need to set some garbage dummy values, so that the documentation can be generated without
    # actually having these values.

    def get_default_architecture(
        input_dimensionality: int, classification: bool = False, n_hidden: int = 500
    ) -> Any:
        raise RuntimeError(_ERROR_MSG)

    def get_architecture_forrester(input_dimensionality: int) -> Any:
        raise RuntimeError(_ERROR_MSG)


logger = get_logger(__name__)


@dataclass
class MetaModelConfig(ABC):
    """Configuration options for the training of the Profet meta-model."""

    benchmark: str
    """ Name of the benchmark. """

    # ---------- "Abstract"/required class attributes:
    json_file_name: ClassVar[str]
    """ Name of the json file that contains the data of this benchmark. """

    get_architecture: ClassVar[
        Callable[[int], "torch.nn.Module"]
    ] = get_default_architecture
    """ Callable that takes the input dimensionality and returns the network to be trained. """

    hidden_space: ClassVar[int]
    """ Size of the hidden space for this benchmark. """

    log_cost: ClassVar[bool]
    """ Whether to apply `numpy.log` onto the raw data for the cost of each point. """

    log_target: ClassVar[bool]
    """ Whether to apply `numpy.log` onto the raw data for the `y` of each point. """

    normalize_targets: ClassVar[bool]
    """ Whether to normalize the targets (y), by default False. """

    shapes: ClassVar[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]
    """ The shapes of the X, Y and C arrays of the dataset. """

    y_min: ClassVar[float]
    """ The minimum of the Y array. """

    y_max: ClassVar[float]
    """ The maximum of the Y array. """

    c_min: ClassVar[float]
    """ The minimum of the C array. """

    c_max: ClassVar[float]
    """ The maximum of the C array. """

    # -----------

    task_id: int = 0
    """ Task index. """

    seed: int = 123
    """ Random seed. """

    num_burnin_steps: int = 50000
    """ (copied from `Bohamiann.train`): Number of burn-in steps to perform. This value is passed
    to the given `optimizer` if it supports special burn-in specific behavior. Networks sampled
    during burn-in are discarded.
    """

    num_steps: int = 13_000
    """Value passed to the argument of the same name in `Bohamiann.train`.

    (copied from `Bohamiann.train`):
    Number of sampling steps to perform after burn-in is finished. In total,
    `num_steps // keep_every` network weights will be sampled.
    """

    mcmc_thining: int = 100
    """ `keep_every` argument of `Bohamiann.train`.

    (copied from `Bohamiann.train`):
    Number of sampling steps (after burn-in) to perform before keeping a sample. In total,
    `num_steps // keep_every` network weights will be sampled.
    """

    lr: float = 1e-2
    """ `lr` argument of `Bohamiann.train`. """

    batch_size: int = 5
    """ `batch_size` argument of `Bohamiann.train`. """

    max_samples: Optional[int] = None
    """ Maximum number of data samples to use when training the meta-model. This can be useful
    if the dataset is large (e.g. FCNet task) and you don't have crazy amounts of memory.
    """

    n_inducing_lvm: int = 50
    """ Passed as the value for the "num_inducing" argument of `BayesianGPLVM` constructor.

    (copied form ``GPy.core.sparse_gp_mpi.SparseGP_MPI``):
    Number of inducing points (optional, default 10. Ignored if Z is not None)
    """

    max_iters: int = 10_000
    """Argument passed to the `optimize` method of the `BayesianGPLVM` instance that is used in the
    call to `get_features`. Appears to be the number of training iterations to perform.
    """

    n_samples_task: int = 500
    """ Number of tasks to create in `get_training_data`."""

    def get_task_network(self, input_path: Union[Path, str]) -> Tuple[Any, np.ndarray]:
        """Create, train and return a surrogate model for the given `benchmark`, `seed` and `task_id`.

        Parameters
        ----------
        input_path : Union[Path, str]
            Data directory containing the json files.

        Returns
        -------
        Tuple[Any, np.ndarray]
            The surrogate model for the objective, as well as an array of sampled task features.
        """
        rng = np.random.RandomState(seed=self.seed)

        X, Y, C = self.load_data(input_path)

        task_features_mean, task_features_std = self._get_features(
            X=X,
            Y=Y,
            C=C,
            display_messages=False,
        )

        X_train, Y_train, C_train = self._get_training_data(
            X,
            Y,
            C,
            task_features_mean=task_features_mean,
            task_features_std=task_features_std,
        )
        objective_model, cost_model = self._get_meta_model(
            X_train,
            Y_train,
            C_train,
            with_cost=False,
        )

        net = self._create_task_network(objective_model, X_train.shape[1])

        multiplier = rng.randn(self.hidden_space)
        h = (
            task_features_mean[self.task_id]
            + task_features_std[self.task_id] * multiplier
        )
        return net, h

    def load_data(
        self, input_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                raise RuntimeError(
                    f"Download finished, but file {file} still doesn't exist!"
                )
        with open(file) as f:
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
            Task indices of corresponding labels Y.

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

    def _get_features(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        C: np.ndarray,
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
            Training costs
        display_messages : bool, optional
            Whether to log messages to the console or not, by default True.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The features mean and std arrays.
        """
        n_tasks = Y.shape[0]
        n_configs = X.shape[0]
        index_task = np.repeat(np.arange(n_tasks), n_configs)
        Y_norm, _, _ = self.normalize_Y(deepcopy(Y.flatten()), index_task)

        # train the probabilistic encoder
        kern = GPy.kern.Matern52(input_dim=self.hidden_space, ARD=True)

        m_lvm = BayesianGPLVM(
            Y_norm.reshape(n_tasks, n_configs),
            input_dim=self.hidden_space,
            kernel=kern,
            num_inducing=self.n_inducing_lvm,
        )
        m_lvm.optimize(max_iters=self.max_iters, messages=display_messages)

        ls = np.array(
            [m_lvm.kern.lengthscale[i] for i in range(m_lvm.kern.lengthscale.shape[0])]
        )

        # generate data to train the multi-task model
        task_features_mean = np.array(m_lvm.X.mean / ls)
        task_features_std = np.array(np.sqrt(m_lvm.X.variance) / ls)

        return task_features_mean, task_features_std

    def _get_training_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        C: np.ndarray,
        task_features_mean: np.ndarray,
        task_features_std: np.ndarray,
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

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            numpy arrays containing the X, Y, and C's for each task.
        """
        n_tasks = Y.shape[0]
        hidden_space = task_features_std.shape[1]
        n_configs = X.shape[0]

        X_train_list = []
        Y_train_list = []
        C_train_list = []

        for i, xi in enumerate(X):
            for idx in range(n_tasks):
                for _ in range(self.n_samples_task):
                    multiplier = np.random.randn(hidden_space)
                    ht = task_features_mean[idx] + task_features_std[idx] * multiplier

                    x = np.concatenate((xi, ht), axis=0)
                    X_train_list.append(x)
                    Y_train_list.append(Y[idx, i])
                    C_train_list.append(C[idx, i])

        X_train = np.array(X_train_list)
        Y_train = np.array(Y_train_list)
        C_train = np.array(C_train_list)
        if self.log_cost:
            C_train = np.log(C_train)

        if self.log_target:
            Y_train = np.log(Y_train)

        return X_train, Y_train, C_train

    def _get_meta_model(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        C_train: np.ndarray,
        with_cost: bool = False,
    ):
        """Create, train and return the objective model, and (optionally) a cost model for the data.

        Parameters
        ----------
        X_train : np.ndarray
            Training samples.
        Y_train : np.ndarray
            Training objectives.
        C_train : np.ndarray
            Training costs.
        with_cost : bool, optional
            Whether to also create a surrogate model for the cost. Defaults to `False`.

        Returns
        -------
        Tuple[Bohamiann, Optional[Bohamiann]]
            Surrogate model for the objective, as well as another for the cost, if `with_cost` is
            True, otherwise `None`.
        """
        objective_model = Bohamiann(
            get_network=type(self).get_architecture,
            print_every_n_steps=1000,
            normalize_output=self.normalize_targets,
        )
        logger.info("Training Bohamiann objective model.")

        if self.max_samples is not None:
            logger.info(
                f"Limiting the dataset to a maximum of {self.max_samples} samples."
            )
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
            cost_model = Bohamiann(
                get_network=type(self).get_architecture, print_every_n_steps=1000
            )
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

    def _create_task_network(self, model, size: int, idx: int = 0) -> "torch.nn.Module":
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
            A module with sampled weights.
        """
        net = model.get_network(size)
        # assert False, (type(self).get_architecture, net, self.shapes)
        with torch.no_grad():
            sampled_weights = model.sampled_weights[idx]
            for parameter, sample in zip(net.parameters(), sampled_weights):
                parameter.copy_(torch.from_numpy(sample))
        return net

    def load_task_network(
        self,
        checkpoint_file: Union[str, Path],
    ) -> Tuple[Any, np.ndarray]:
        """Load the result of the `get_task_network` function stored in the pickle file.

        Parameters
        ----------
        checkpoint_file : Union[str, Path]
            Path to a pickle file. The file is expected to contain a serialized dictionary, with keys
            "benchmark", "size", "network", and "h".

        Returns
        -------
        Tuple[Any, np.ndarray]
            The surrogate model for the objective, as well as an array of sampled task features.
        """
        with open(checkpoint_file, "rb") as f:
            state = pickle.load(f)

        if state["benchmark"] != self.benchmark:
            raise RuntimeError(
                f"Trying to load model for benchmark {self.benchmark} from checkpoint that "
                f"contains data from benchmark {state['benchmark']}."
            )
        network = type(self).get_architecture(input_dimensionality=state["size"])
        network.load_state_dict(state["network"])
        h = state["h"]

        return network, h

    def save_task_network(
        self, checkpoint_file: Union[str, Path], network: Any, h: np.ndarray
    ) -> None:
        """Save the meta-model for the task at the given path.

        Parameters
        ----------
        checkpoint_file : Union[str, Path]
            Path where the model should be saved
        network : Any
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
