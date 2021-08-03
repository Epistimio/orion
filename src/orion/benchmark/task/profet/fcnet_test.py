from pathlib import Path
import pytest
from .fcnet import FcNetTask

input_path: Path = Path("profet_data/data")


@pytest.mark.skipif(not input_path.exists(), reason="test requires profet data")
def test_kwargs_fix_dimension_fcnet():
    # TODO: Test that setting 'kwargs' actually fixes a dimension.
    task = FcNetTask(input_path=input_path, batch_size=53)
    xs = task.sample()
    assert len(xs) == 1
    x = xs[0]
    assert x.batch_size == 53

    xs = task.sample(n=10)
    assert len(xs) == 10
    assert xs and all(x.batch_size == 53 for x in xs)
