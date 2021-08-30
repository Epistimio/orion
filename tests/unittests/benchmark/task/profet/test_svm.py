import pytest

from orion.benchmark.task.profet.svm import SvmTask
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig


@pytest.mark.skip(reason="Take WAY too long to run.")
def test_kwargs_fix_dimension_svm(profet_train_config: MetaModelTrainingConfig):
    # TODO: Test that setting 'kwargs' actually fixes a dimension.
    task = SvmTask(gamma=0.123, train_config=profet_train_config)
    xs = task.sample()
    assert len(xs) == 1
    x = xs[0]
    assert x.gamma == 0.123

    xs = task.sample(n=10)
    assert len(xs) == 10
    assert xs and all(x.gamma == 0.123 for x in xs)
