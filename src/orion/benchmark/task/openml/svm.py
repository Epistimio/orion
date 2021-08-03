""" SVM task from OpenML used in the ABLR paper, as described in Section B.2 of the
supplementary material.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, Type, Union

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from simple_parsing.helpers.hparams import HyperParameters, categorical, uniform

import openml

from .openml_task import OpenMLTask, OpenMLTaskHParams

FLOW_ID = 5891  # PAPER
# FLOW_ID = 8817
# " we considered the 30 most evaluated data sets whose task ids are"
TASK_IDS = [
    10101,
    145878,
    146064,
    14951,
    # 34536, # having issues with this ID.
    34537,
    3485,
    3492,
    3493,
    3494,
    37,
    3889,
    3891,
    3899,
    3902,
    3903,
    3913,
    3918,
    3950,
    # 6566, # having issues with this ID.
    9889,
    9914,
    9946,
    9952,
    9967,
    9971,
    9976,
    9978,
    9980,
    9983,
]


@dataclass
class SvmTaskHParams(OpenMLTaskHParams):
    cost: float = uniform(0.000986, 998.492437, default=500)
    degree: int = uniform(2, 5, default=3, discrete=True)
    gamma: float = uniform(0.000988, 913.373845, default=500)
    kernel: str = categorical(
        # # From the paper
        # "linear", "polynomial", "radial", "sigmoid", default="linear"
        "linear",
        "poly",
        "rbf",
        "sigmoid",
        "precomputed",
        default="rbf",
    )


class SvmTask(OpenMLTask):
    hparams: ClassVar[Type[HyperParameters]] = SvmTaskHParams

    def __init__(
        self,
        task_id: int = TASK_IDS[0],
        flow_id: int = FLOW_ID,
        rng: np.random.RandomState = None,
        seed: int = None,
        **fixed_dims
    ):
        if task_id not in TASK_IDS:
            # Treat the task_id as an 'index', rather than an 'identifier'.
            task_id = TASK_IDS[task_id]
        super().__init__(task_id=task_id, flow_id=flow_id, seed=seed)

    def __call__(
        self, hp: Union[HyperParameters, Dict, np.ndarray] = None, **kwargs
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
        if hp is None and kwargs:
            hp = self.hparams(**kwargs)
        hparams: SvmTaskHParams
        if isinstance(hp, self.hparams):
            hparams = hp
        elif isinstance(hp, dict):
            hparams = self.hparams(**hp)
        elif isinstance(hp, np.ndarray):
            hparams = self.hparams.from_array(hp)
        elif isinstance(hp, list):
            return list(map(self, hp))
        else:
            raise NotImplementedError(hp)

        # TODO: Getting this 'RuntimeError: No extension could be found for flow 5891: mlr.classif.svm',
        # which seems to indicate that the flow needs to be ran in the R programming language!
        # For now, I'll try to use the closest flow, that uses sklearn instead.
        # TODO: Will also need to give a mapping from their hparams to the hparams of that new flow.
        # TODO: Need to figure out how to pass new hparams to a given flow.
        # from sklearn.svm import SVC
        # TODO: Some values don't correspond
        svm = SVC(
            C=hparams.cost,
            degree=hparams.degree,
            kernel=hparams.kernel,
            gamma=hparams.gamma,
        )

        model = Pipeline(
            steps=[("transform", SimpleImputer(strategy="median")), ("estimator", svm),]
        )
        run = openml.runs.run_model_on_task(model, task=self.task, seed=self.seed)

        # TODO: How to extract the 'performance' from this 'run'?
        scores = run.get_metric_fn(accuracy_score)
        score = np.mean(scores, -1)

        return score
        # run.open_in_browser()
        return run.task_evaluation_measure


if __name__ == "__main__":
    task = SvmTask(task_id=2)
    # x = task.hparams()
    dataset = task.make_dataset_np(100)
    # x = task.sample()[0]
    # y = task(x)
    # print(x)
    # print(y)
    print(dataset)
