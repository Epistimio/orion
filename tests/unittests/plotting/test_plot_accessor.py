import pytest

from orion.plotting import PlotAccessor

def test_init_require_experiment():
    with pytest.raises(ValueError):
        PlotAccessor(None)

def test_instance_call_defined():
    pass # TODO

def test_regret():
    pass # TODO