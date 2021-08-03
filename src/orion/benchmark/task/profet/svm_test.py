from pathlib import Path
import pytest
from .svm import SvmTask

input_path: Path = Path("profet_data/data")


@pytest.mark.skipif(not input_path.exists(), reason="test requires profet data")
def test_kwargs_fix_dimension_svm():
    # TODO: Test that setting 'kwargs' actually fixes a dimension.
    task = SvmTask(input_path=input_path, gamma=0.123)
    xs = task.sample()
    assert len(xs) == 1
    x = xs[0]
    assert x.gamma == 0.123

    xs = task.sample(n=10)
    assert len(xs) == 10
    assert xs and all(x.gamma == 0.123 for x in xs)
