from pathlib import Path
import pytest
from .svm import SvmTask
from .profet_task import MetaModelTrainingConfig

@pytest.mark.skip(reason="Take WAY too long to run.")
def test_kwargs_fix_dimension_svm(profet_train_config: MetaModelTrainingConfig):
    # TODO: Test that setting 'kwargs' actually fixes a dimension.
    task = SvmTask(gamma=0.123, get_task_network_kwargs={})
    xs = task.sample()
    assert len(xs) == 1
    x = xs[0]
    assert x.gamma == 0.123

    xs = task.sample(n=10)
    assert len(xs) == 10
    assert xs and all(x.gamma == 0.123 for x in xs)
