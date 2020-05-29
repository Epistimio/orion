import pytest

from orion.plotting import PlotAccessor
from orion.core.utils.tests import create_experiment


config = dict(
    name='experiment-name',
    space={'x': 'uniform(0, 200)'},
    metadata={'user': 'test-user',
              'orion_version': 'XYZ',
              'VCS': {"type": "git",
                      "is_dirty": False,
                      "HEAD_sha": "test",
                      "active_branch": None,
                      "diff_sha": "diff"}},
    version=1,
    pool_size=1,
    max_trials=10,
    working_dir='',
    algorithms={'random': {'seed': 1}},
    producer={'strategy': 'NoParallelStrategy'},
)


trial_config = {
    'experiment': 0,
    'status': 'new',  # new, reserved, suspended, completed, broken
    'worker': None,
    'start_time': None,
    'end_time': None,
    'heartbeat': None,
    'results': [],
    'params': []
}


def test_init_require_experiment():
    with pytest.raises(ValueError):
        PlotAccessor(None)


def test_instance_call_defined():
    with create_experiment(config, trial_config) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        with pytest.raises(NotImplementedError):
            pa()


def test_regret():
    with create_experiment(config, trial_config) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        with pytest.raises(NotImplementedError):
            pa.regret()
