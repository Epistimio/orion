import pytest
import plotly

from orion.plotting import regret
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
    'status': 'completed',
    'worker': None,
    'start_time': None,
    'end_time': None,
    'heartbeat': None,
    'results': [],
    'params': []
}


def test_requires_argument():
    with pytest.raises(ValueError):
        regret(None)


def test_returns_plotly_object():
    with create_experiment(config, trial_config, ['completed']) as (_, _, experiment):
        plot = regret(experiment)

    assert type(plot) is plotly.graph_objects.Figure


def test_basic_graph_layout():
    with create_experiment(config, trial_config, ['completed']) as (_, _, experiment):
        plot = regret(experiment)

    assert plot.layout.title.text == "Regret for experiment 'experiment-name'"
    assert plot.layout.xaxis.title.text == "Trials"
    assert plot.layout.yaxis.title.text == "Performance"

    trace1 = plot.data[0]
    assert trace1.type == "scatter"
    assert trace1.name == "objective"
    assert trace1.mode == "markers"
    assert len(trace1.y) == 1
    assert not trace1.x

    trace2 = plot.data[1]
    assert trace2.type == "scatter"
    assert trace2.name == "best"
    assert trace2.mode == "lines"
    assert len(trace2.y) == 1
    assert not trace2.x
