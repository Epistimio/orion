from pathlib import Path
import pytest
from orion.benchmark.task.profet.xgboost import XgBoostTask
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig


@pytest.mark.skip(reason="Take WAY too long to run.")
def test_kwargs_fix_dimension_xgboost(profet_train_config: MetaModelTrainingConfig):
    # TODO: Test that setting 'kwargs' actually fixes a dimension.
    task = XgBoostTask(benchmark="xgboost", nb_estimators=251, train_config=profet_train_config)
    points = task.sample()
    assert len(points) == 1
    x = points[0]
    assert x.nb_estimators == 251

    points = task.sample(n=10)
    assert len(points) == 10
    assert points and all(x.nb_estimators == 251 for x in points)
