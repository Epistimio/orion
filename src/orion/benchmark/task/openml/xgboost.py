""" XGBoost task from OpenML used in the ABLR paper, as described in Section B.2 of the
supplementary material.
"""
from dataclasses import dataclass
from typing import ClassVar, Dict, Type, Union

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from simple_parsing.helpers.hparams import HyperParameters, categorical, uniform

import openml

from .openml_task import OpenMLTask, OpenMLTaskHParams

FLOW_ID = 6767  # PAPER
# Trying out different flow IDs.
# FLOW_ID = 17488
# FLOW_ID = 1849
# FLOW_ID = 5521
# " we considered the 30 most evaluated data sets whose task ids are"
TASK_IDS = [
    10093,
    10101,
    125923,
    145847,
    145857,
    145862,
    145872,
    145878,
    145953,
    145972,
    145976,
    145979,
    146064,
    14951,
    31,
    3485,
    3492,
    3493,
    37,
    3896,
    3903,
    3913,
    3917,
    3918,
    3,
    49,
    9914,
    9946,
    9952,
    9967,
]


@dataclass
class XGBoostTaskHParams(OpenMLTaskHParams):
    alpha: float = uniform(0.000985, 1009.209690)
    booster: str = categorical("gbtree", "gblinear")
    colsample_bylevel: float = uniform(0.046776, 0.998424)
    colsample_bytree: float = uniform(0.062528, 0.999640)
    eta: float = uniform(0.000979, 0.995686)
    # note: not called "lambda" because that's a reserved keyword in python.
    lamda: float = uniform(0.000978, 999.020893)
    max_depth: int = uniform(1, 15, discrete=True)
    min_child_weight: float = uniform(1.012169, 127.041806)
    nrounds: int = uniform(3, 5000, discrete=True)
    subsample: float = uniform(0.100215, 0.999830)


class XGBoostTask(OpenMLTask):
    hparams: ClassVar[Type[HyperParameters]] = XGBoostTaskHParams

    def __init__(
        self,
        task_id: int = TASK_IDS[0],
        flow_id: int = FLOW_ID,
        rng: np.random.RandomState = None,
        seed: int = None,
        **fixed_dims
    ):
        # TODO: This could be ambiguous for the task indentifier `3` in the list
        # above.
        if task_id not in TASK_IDS:
            # Treat the task_id as an 'index', rather than an 'identifier'.
            task_id = TASK_IDS[task_id]
        super().__init__(task_id=task_id, flow_id=flow_id, rng=rng)

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
        if isinstance(hp, self.hparams):
            hparams = hp
        elif isinstance(hp, dict):
            hparams = self.hparams(**hp)
        elif isinstance(hp, np.ndarray):
            hparams = self.hparams.from_array(hp)
        else:
            raise NotImplementedError(hp)
        assert isinstance(hparams, self.hparams)
        hparams: XGBoostTaskHParams
        from sklearn.preprocessing import StandardScaler

        # Corresponding values for XGBClassifier from xgboost package:
        # - [X] alpha
        # - [X] booster
        # - [X] colsample_bylevel
        # - [X] colsample_bytree
        # - [X] eta
        # - [X] lamda
        # - [X] max_depth
        # - [X] min_child_weight
        # - [ ] nrounds
        # - [X] subsample
        from xgboost.sklearn import XGBClassifier

        # Hparams from the paper:
        # Corresponding values in GradientBoostingClassifier:
        # - [ ] alpha
        # - [ ] booster
        # - [ ] colsample_bylevel
        # - [ ] colsample_bytree
        # - [ ] eta
        # - [ ] lamda
        # - [X] max_depth
        # - [ ] min_child_weight
        # - [ ] nrounds
        # - [ ] subsample
        # sklearn_model = GradientBoostingClassifier(
        #     max_depth=hparams.max_depth, # We have this one.
        #     loss="deviance",
        #     learning_rate=0.1,
        #     n_estimators=100,
        #     subsample=1.0,
        #     criterion="friedman_mse",
        #     min_samples_split=2,
        #     min_samples_leaf=1,
        #     min_weight_fraction_leaf=0.0,
        #     min_impurity_decrease=0.0,
        #     min_impurity_split=None,
        #     init=None,
        #     random_state=None,
        #     max_features=None,
        #     verbose=0,
        #     max_leaf_nodes=None,
        #     warm_start=False,
        #     presort="auto",
        #     validation_fraction=0.1,
        #     n_iter_no_change=None,
        #     tol=0.0001,
        # )

        # Corresponding values in GradientBoostingRegressor:
        # - [ ] alpha
        # - [ ] booster
        # - [ ] colsample_bylevel
        # - [ ] colsample_bytree
        # - [ ] eta
        # - [ ] lamda
        # - [X] max_depth
        # - [ ] min_child_weight
        # - [ ] nrounds
        # - [X] subsample
        # sklearn_model = GradientBoostingRegressor(
        #     max_depth=hparams.max_depth,
        #     subsample=hparams.subsample,
        #     loss="deviance",
        #     learning_rate=0.1,
        #     n_estimators=100,
        #     criterion="friedman_mse",
        #     min_samples_split=2,
        #     min_samples_leaf=1,
        #     min_weight_fraction_leaf=0.0,
        #     min_impurity_decrease=0.0,
        #     min_impurity_split=None,
        #     init=None,
        #     random_state=None,
        #     max_features=None,
        #     verbose=0,
        #     max_leaf_nodes=None,
        #     warm_start=False,
        #     presort="auto",
        #     validation_fraction=0.1,
        #     n_iter_no_change=None,
        #     tol=1e-4,
        # )

        # FROM https://xgboost.readthedocs.io/en/latest/python/python_api.html
        sklearn_model = XGBClassifier(
            booster=hparams.booster,
            max_depth=hparams.max_depth,
            reg_alpha=hparams.alpha,
            reg_lambda=hparams.lamda,
            colsample_bylevel=hparams.colsample_bylevel,
            colsample_bytree=hparams.colsample_bytree,
            min_child_weight=hparams.min_child_weight,
            subsample=hparams.subsample,
            learning_rate=hparams.eta,  # ETA :== boosting learning rate.
        )
        # In case we want to create our own model:
        model = make_pipeline(SimpleImputer(), StandardScaler(), sklearn_model)
        run = openml.runs.run_model_on_task(model, task=self.task, seed=self.seed)
        # TODO: How to extract the 'performance' from this 'run'?
        scores = run.get_metric_fn(accuracy_score)
        score = np.mean(scores, -1)
        return score


if __name__ == "__main__":
    task = XGBoostTask()
    x = task.sample(1)
    y = task(x[0])
    print(x)
    print(y)
