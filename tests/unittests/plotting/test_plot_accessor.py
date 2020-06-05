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


def check_regret_plot(plot):
    assert plot
    assert "regret" in plot.layout.title.text.lower()
    assert 2 == len(plot.data)


def test_init_require_experiment():
    with pytest.raises(ValueError):
        PlotAccessor(None)


def test_call_nonexistent_kind():
    with create_experiment(config, trial_config, ['completed']) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        with pytest.raises(KeyError):
            pa(kind='nonexistent')


def test_regret_is_default_plot():
    with create_experiment(config, trial_config, ['completed']) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        plot = pa()

        check_regret_plot(plot)


def test_regret_kind():
    with create_experiment(config, trial_config, ['completed']) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        plot = pa(kind='regret')

        check_regret_plot(plot)


def test_call_to_regret():
    with create_experiment(config, trial_config, ['completed']) as (_, _, experiment):
        pa = PlotAccessor(experiment)
        plot = pa.regret()

        check_regret_plot(plot)
