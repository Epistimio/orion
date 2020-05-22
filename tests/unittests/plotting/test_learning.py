import pytest
from orion.plotting import learning

def test_learning_requires_argument():
    with pytest.raises(ValueError):
        learning(None)