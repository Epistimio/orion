from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Generic,
    List,
    NamedTuple,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import hashlib

import numpy as np
import torch
from orion.benchmark.task.base import BaseTask
from simple_parsing import Serializable
from torch import Tensor
from torch.utils.data import TensorDataset
from simple_parsing.helpers.hparams import HyperParameters
from orion.benchmark.task.utils import dict_union
from logging import getLogger as get_logger


HParams = TypeVar("HParams", bound=HyperParameters)
logger = get_logger(__name__)


class Domain(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray

    def __contains__(self, item: Union[np.ndarray, Tensor]) -> bool:
        if isinstance(item, Tensor):
            item = item.cpu().numpy()
        assert all(self.lower <= self.upper), "lower > upper?"
        if not item.shape[-1] == self.lower.shape[-1]:
            return False
        return np.all(self.lower <= item) and np.all(item <= self.upper)


class Task(BaseTask, Generic[HParams], ABC):
    """ Simple ABC for a 'Task'.

    Can evaluate dataclasses and dicts, and can have some input dimensions be fixed by
    passing them as keyword arguments to the constructor.
    """

    hparams: ClassVar[Type[HParams]]

    def __init__(
        self,
        max_trials: int = 100,
        task_id: int = 0,
        rng: np.random.RandomState = None,
        seed: int = None,
        fixed_dims: Dict[str, Any] = None,
    ):
        super().__init__(max_trials=max_trials)
        self.task_id = task_id
        self.rng = rng or np.random.RandomState(seed)
        self.seed = seed
        fixed_dims = fixed_dims or {}
        # Fixed dimensions in the hparam space.
        self.fixed_values: Dict[str, Any] = {
            key: value
            for key, value in fixed_dims.items()
            if key in self.hparams.get_orion_space_dict()
        }
        logger.debug(
            f"Creating a Task of type {type(self).__name__}, with task id "
            f"{task_id} and with fixed values {self.fixed_values}"
        )

    @abstractmethod
    def __call__(
        self, hp: Union[HParams, Dict, np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """Evaluates the given samples and returns the performance.

        Args:
            hp (Union[HyperParameters, Dict, np.ndarray], optional):
                Either a Hyperparameter dataclass, a dictionary, or a numpy
                array containing the values for each dimension. Defaults to
                None.

        Returns:
            np.ndarray: The performances of the hyperparameter on the sampled
            task.
        """

        # return super().__call__(**kwargs)

    def call(self, *args, **kwargs):
        return self(*args, **kwargs)

    @property
    def max_trials(self) -> int:
        """Return the max number of trials to run with this objective."""
        return self._max_trials

    def get_search_space(self) -> Dict[str, str]:
        """Return the search space for the task objective function"""
        return self.space

    @property
    def configuration(self):
        """Return the configuration of the task."""
        return {
            self.__class__.__qualname__: {
                "task_id": self.task_id,
                "seed": self.seed,
                "fixed_values": self.fixed_values,
                "max_trials": self.max_trials,
            }
        }

    @property
    def full_space(self) -> Dict[str, str]:
        """ Auto-generated orion-style 'space' dict for `self.hparams`.

        This is generated using the `self.hparams` dataclass type.

        Returns:
            Dict[str, str]: Dict mapping from dimension name to the 'space'.
        """
        return self.hparams.get_orion_space_dict()

    @classmethod
    def get_domain(cls) -> Domain:
        bounds = cls.hparams.get_bounds()
        return Domain(
            lower=np.asfarray([bound.domain[0] for bound in bounds]),
            upper=np.asfarray([bound.domain[1] for bound in bounds]),
        )

    @property
    def space(self) -> Dict[str, str]:
        """ Auto-generated orion-style 'space' strings for `self.hparams`.

        Entries from `self.fixed_values` are removed from the returned dictionary.
        """
        space = self.full_space.copy()
        for k in self.fixed_values:
            space.pop(k)
        return space

    @property
    def full_domain(self) -> List[Dict[str, Any]]:
        """ Auto-generated GPyOpt-style 'domain' dicts for `self.hparams`. """
        return self.hparams.get_bounds_dicts()

    @property
    def domain(self) -> List[Dict[str, Any]]:
        """ Auto-generated GPyOpt-style 'domain' dicts for `self.hparams`.

        Entries with a name in `self.fixed_values` are not returned.
        """
        domains = self.full_domain.copy()
        return list(filter(lambda d: d["name"] not in self.fixed_values, domains))

    def seed_rng(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hparams.rng = self.rng

    def sample(self, n: int = 1) -> List[HParams]:
        """Samples `n` random hyperparameter examples from the priors.

        If any dimensions are 'fixed' (using `self.fixed_values`), the returned
        samples have the corresponding dimensions set to the value in
        `self.fixed_values`.

        Args:
            n (int, optional): Number of samples. Defaults to 1.

        Returns:
            List[HyperParameters]: List of 'random' `HyperParameters` objects.
        """
        self.hparams.rng = self.rng
        samples = [self.hparams.sample() for i in range(n)]
        for item in samples:
            # If there are 'fixed' values in `self.fixed_values`, set them on the
            # sampled objects.
            for k, v in self.fixed_values.items():
                if hasattr(item, k):
                    setattr(item, k, v)
        return samples

    def sample_np(self, n: int) -> np.ndarray:
        hps: List[HyperParameters] = self.sample(n)
        x = np.asarray([hp.to_array() for hp in hps], dtype=np.float32)
        return x

    # @property
    # def hparams(self) -> Type[HyperParameters]:
    #     # NOTE: This might be a bit slow (because of the imports), so it's
    #     # better to use a concrete subclass of `ProfetTask` like SvmTask or
    #     # FcNetTask, etc.
    #     if self.benchmark == "svm":
    #         from . import SvmTaskHParams
    #         return SvmTaskHParams
    #     elif self.benchmark == "forrester":
    #         from . import ForresterTaskHParams
    #         return ForresterTaskHParams
    #     elif self.benchmark == "fcnet":
    #         from . import FcNetTaskHParams
    #         return FcNetTaskHParams
    #     elif self.benchmark == "xgboost":
    #         from . import XgBoostTaskHParams
    #         return XgBoostTaskHParams
    #     else:
    #         raise RuntimeError(
    #             f"Benchmark {self.benchmark} doesn't have a registered hparams class! :(")

    def make_dataset(self, n_samples: int) -> TensorDataset:
        """ Create a Dataset containing hps and performances from `task`. """

        x, y = self.make_dataset_np(n_samples)
        x = torch.as_tensor(x, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.float)
        dataset = TensorDataset(x, y)
        return dataset

    def make_dataset_np(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Create a Dataset containing hps and performances from `task`. """
        hps = self.sample(n_samples)
        perfs = self(hps)
        x = np.asarray([hp.to_array() for hp in hps], dtype=np.float32)
        y = np.asarray(perfs, dtype=np.float32)
        y = y.reshape([-1])
        return x, y

    def to_hp_dict(self, hp: Union[HParams, Dict, np.ndarray], **kwargs) -> Dict:
        """ Convert a given hyper-parameter to a dict of hparam_name: value.
        
        Values from `kwargs` are used as 'fixed' dimension values, which will
        overwrite those from `hp` in the returned dictionary.
        """
        keys = set(self.full_space.keys())
        if isinstance(hp, np.ndarray):
            if hp.ndim == 1:
                hp = dict(zip(keys, hp))
            elif hp.ndim == 2 and hp.shape[0] == 1:
                hp = dict(zip(keys, hp[0]))
            else:
                assert False, "hp has too many dims!"

        if isinstance(hp, list):
            assert len(hp) == 1, hp
            hp = hp[0]
            assert isinstance(hp, self.hparams)

            # hp = dict(zip(keys, hp))
            # # If this function is passed a list of hyperparameters, we
            # # recursively call `self` for each element and return a float array.
            # return np.asfarray(list(map(self, hp)))

        if isinstance(hp, Serializable):
            # Convert the Hparams to a dict, and overwrite with kwargs if given.
            kwargs = dict_union(hp.to_dict(), kwargs)
        elif is_dataclass(hp):
            kwargs = dict_union(asdict(hp), kwargs)
        elif isinstance(hp, dict):
            kwargs = dict_union(hp, kwargs)
        return kwargs


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

