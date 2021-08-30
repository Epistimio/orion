import os
from pathlib import Path

import pytest

from orion.benchmark.task.profet.fcnet import FcNetTask
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig


@pytest.mark.skip(reason="Take WAY too long to run.")
def test_kwargs_fix_dimension_fcnet(profet_train_config: MetaModelTrainingConfig):
    """ Test that setting 'kwargs' actually fixes a dimension. """
    task = FcNetTask(batch_size=53, train_config=profet_train_config)
    xs = task.sample()
    assert len(xs) == 1
    x = xs[0]
    assert x.batch_size == 53

    xs = task.sample(n=10)
    assert len(xs) == 10
    assert xs and all(x.batch_size == 53 for x in xs)
