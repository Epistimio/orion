""" TODO: Add the tasks from OpenML used in the ABLR paper, section 4.3
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from orion.benchmark.task.task import Task
from simple_parsing.helpers.hparams import HyperParameters
from logging import getLogger as get_logger
import openml

logger = get_logger(__name__)


@dataclass
class OpenMLTaskHParams(HyperParameters):
    pass


class OpenMLTask(Task, ABC):
    hparams: ClassVar[Type[HyperParameters]] = OpenMLTaskHParams

    def __init__(
        self,
        task_id: int,
        flow_id: int,
        rng: np.random.RandomState = None,
        seed: int = None,
        **fixed_dims,
    ):
        super().__init__(task_id=task_id, rng=rng, seed=seed, **fixed_dims)
        logger.info(f"Creating new openml task for task {task_id} and flow {flow_id}")
        self.task = openml.tasks.get_task(task_id, download_data=True)
        self.flow_id = flow_id
        self.flow = openml.flows.get_flow(flow_id, strict_version=False)
        # openml.extensions.sklearn.SklearnExtension().flow_to_model(self.flow, strict_version=False)

    @abstractmethod
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
        # run = openml.runs.run_model_on_task(model, task=self.task, seed=self.seed)
        run = openml.runs.run_flow_on_task(self.flow, task=self.task, seed=self.seed)
        # TODO: How to extract the 'performance' from this 'run'?
        scores = run.get_metric_fn(accuracy_score)
        score = np.mean(scores, -1)
        return score

    @property
    def flow_type(self) -> str:
        return self.flow.name.split(".")[-1]

    def make_dataset_np(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """ Create a Dataset containing hps and performances from `task`. """

        eval_df = openml.evaluations.list_evaluations_setups(
            function="predictive_accuracy",
            tasks=[self.task.id],
            flows=[self.flow.id],
            uploaders=[2702],
            output_format="dataframe",
            parameters_in_separate_columns=True,
            # NOTE: Get *ALL* evaluations, in hopes that some won't have missing values.
            # size=n_samples,
        )
        flow_type = self.flow_type
        # TODO: Maybe select rows without nans?
        hparam_df, performances = self.create_table_from_evaluations(
            eval_df=eval_df, run_count=n_samples,
        )
        return hparam_df.to_numpy(), performances.to_numpy()

    def make_dataset(self, n_samples: int) -> Dataset:
        metric = "predictive_accuracy"
        # # TODO: Use functions from https://github.com/openml/openml-python/blob/develop/examples/40_paper/2018_neurips_perrone_example.py
        # # to download data from these flows / task_id combinations.
        # evals = openml.evaluations.list_evaluations_setups(
        #     function=metric,
        #     tasks=[self.task.id],
        #     flows=[self.flow.id],
        #     size=1,
        #     output_format="dataframe",
        # )
        hparam_arrays, perfs = self.make_dataset_np(n_samples=n_samples)
        param_names = self.hparams.field_names()

        # TODO: Use some sort of sklearn inputer to fill missing values.
        import torch

        x = torch.as_tensor(x, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.float)

        hparam_dicts = list(dict(zip(param_names, v)) for v in hparam_arrays)
        dataset = list(zip(hparam_dicts, perfs))
        return dataset

    def create_table_from_evaluations(
        self,
        eval_df: pd.DataFrame,
        run_count: int = np.iinfo(np.int64).max,
        task_ids: List[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a tabular data with its ground truth from a dataframe of evaluations.
        Optionally, can filter out records based on task ids.

        Parameters
        ----------
        eval_df : dataframe
            Containing list of runs as obtained from list_evaluations()
        flow_type : str, {'svm', 'xgboost'}
            To select whether svm or xgboost experiments are to be run
        run_count : int
            Maximum size of the table created, or number of runs included in the table
        task_ids : list, (optional)
            List of integers specifying the tasks to be retained from the evaluations dataframe

        Returns
        -------
        eval_table : dataframe
        values : list
        """
        if task_ids is not None:
            eval_df = eval_df[eval_df["task_id"].isin(task_ids)]
        column_names: List[str] = [f.name for f in fields(self.hparams)]

        eval_df = eval_df.sample(frac=1)  # shuffling rows
        eval_df.columns = [column.split("_")[-1] for column in eval_df.columns]

        # eval_df = eval_df.convert_dtypes()
        # eval_df.fillna(value=np.nan, inplace=True)
        # TODO: try to get the 'cleanest' `run_count` runs (without n/a values, if possible)
        # pd.to_numeric(eval_df, errors="ignore")

        # IDEA: Use the type annotations to select the right dtype to convert to in
        # pandas.
        field_types = {f.name: f.type for f in fields(self.hparams)}
        # Dict mapping from type, to kwargs of the `to_numeric` function.
        types_to_to_numeric_kwargs: Dict[Type, Union[bool, str]] = {
            int: {"downcast": "integer", "errors": "coerce"},
            float: {"downcast": "float", "errors": "raise"},
        }
        for field_name, field_type in field_types.items():
            column = eval_df[field_name]
            to_numeric_kwargs = types_to_to_numeric_kwargs.get(field_type)

            if to_numeric_kwargs:
                eval_df[field_name] = pd.to_numeric(column, **to_numeric_kwargs)
            else:
                eval_df[field_name] = column.convert_dtypes()

        # Sort the dataframe by the number of nan values per row (fewer nan's first).
        eval_df = eval_df.iloc[eval_df.isna().sum(axis=1).argsort()]

        for field_name, field_type in field_types.items():
            column = eval_df[field_name]
            # IDEA: Choose how to fill missing values depending on the type.
            if field_type is str:
                # Weird bug in pandas, fillna doesn't do anything with string columns.
                column.fillna(column.mode()[0], inplace=True)
            else:
                column.fillna(column.mean(), inplace=True)

        # Only take the first `run_count` values.
        # NOTE: We sorted by increasing number of nan values above, so we prioritize
        # 'clean' data.
        eval_df = eval_df.iloc[:run_count, :]
        # Take the columns for the hparams of interest.
        hparam_df: pd.DataFrame = eval_df.loc[:, column_names]
        value: pd.Series = eval_df.loc[:, "value"]
        return hparam_df, value

    @property
    def hash(self) -> str:
        # TODO: Return a unique "hash"/id/key for this task
        from orion.benchmark.task.utils import compute_identity
        from dataclasses import is_dataclass, asdict

        if is_dataclass(self):
            return compute_identity(**asdict(self))
        return compute_identity(
            task_id=self.task_id,
            flow_id=self.flow_id,
            **self.fixed_values,
            seed=self.seed,
        )
