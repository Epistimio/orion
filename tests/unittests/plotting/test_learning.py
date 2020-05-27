import pytest
from orion.plotting import regret

def test_regret_requires_argument():
    with pytest.raises(ValueError):
        regret(None)