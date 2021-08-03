from pathlib import Path
import pytest
from .xgboost import XgBoostTask

input_path: Path = Path("profet_data/data")


@pytest.mark.skipif(not input_path.exists(), reason="test requires profet data")
def test_kwargs_fix_dimension_xgboost():
    # TODO: Test that setting 'kwargs' actually fixes a dimension.
    task = XgBoostTask(input_path=input_path, benchmark="xgboost", nb_estimators=251,)
    points = task.sample()
    assert len(points) == 1
    x = points[0]
    assert x.nb_estimators == 251

    points = task.sample(n=10)
    assert len(points) == 10
    assert points and all(x.nb_estimators == 251 for x in points)
