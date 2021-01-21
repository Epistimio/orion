"""Collection of tests for :mod:`orion.plotting.backend_plotly`."""
import copy

import numpy
import pandas
import plotly
import pytest

import orion.client
from orion.core.worker.experiment import Experiment, ExperimentView
from orion.plotting.base import lpi, parallel_coordinates, rankings, regret, regrets
from orion.testing import create_experiment

config = dict(
    name="experiment-name",
    space={"x": "uniform(0, 200)"},
    metadata={
        "user": "test-user",
        "orion_version": "XYZ",
        "VCS": {
            "type": "git",
            "is_dirty": False,
            "HEAD_sha": "test",
            "active_branch": None,
            "diff_sha": "diff",
        },
    },
    version=1,
    pool_size=1,
    max_trials=10,
    working_dir="",
    algorithms={"random": {"seed": 1}},
    producer={"strategy": "NoParallelStrategy"},
)

trial_config = {
    "experiment": 0,
    "status": "completed",
    "worker": None,
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [],
    "params": [],
}


def assert_rankings_plot(plot, names, balanced=10, with_avg=False):
    """Checks the layout of a rankings plot"""
    assert plot.layout.title.text == "Average Rankings"
    assert plot.layout.xaxis.title.text == "Trials ordered by suggested time"
    assert plot.layout.yaxis.title.text == "Ranking based on loss"

    if with_avg:
        line_plots = plot.data[::2]
        err_plots = plot.data[1::2]
    else:
        line_plots = plot.data

    assert len(line_plots) == len(names)

    for name, trace in zip(names, line_plots):
        assert trace.type == "scatter"
        assert trace.name == name
        assert trace.mode == "lines"
        if balanced:
            assert len(trace.y) == balanced
            assert len(trace.x) == balanced

    if with_avg:
        assert len(err_plots) == len(names)
        for name, trace in zip(names, err_plots):
            assert trace.fill == "toself"
            assert trace.name == name
            assert not trace.showlegend
            if balanced:
                assert len(trace.y) == balanced * 2
                assert len(trace.x) == balanced * 2


def assert_regret_plot(plot):
    """Checks the layout of a regret plot"""
    assert plot.layout.title.text == "Regret for experiment 'experiment-name'"
    assert plot.layout.xaxis.title.text == "Trials ordered by suggested time"
    assert plot.layout.yaxis.title.text == "Objective 'loss'"

    trace1 = plot.data[0]
    assert trace1.type == "scatter"
    assert trace1.name == "trials"
    assert trace1.mode == "markers"
    assert len(trace1.y) == 1
    assert not trace1.x

    trace2 = plot.data[1]
    assert trace2.type == "scatter"
    assert trace2.name == "best-to-date"
    assert trace2.mode == "lines"
    assert len(trace2.y) == 1
    assert not trace2.x


def assert_regrets_plot(plot, names, balanced=10, with_avg=False):
    """Checks the layout of a regrets plot"""
    assert plot.layout.title.text == "Average Regret"
    assert plot.layout.xaxis.title.text == "Trials ordered by suggested time"
    assert plot.layout.yaxis.title.text == "loss"

    if with_avg:
        line_plots = plot.data[::2]
        err_plots = plot.data[1::2]
    else:
        line_plots = plot.data

    assert len(line_plots) == len(names)

    for name, trace in zip(names, line_plots):
        assert trace.type == "scatter"
        assert trace.name == name
        assert trace.mode == "lines"
        if balanced:
            assert len(trace.y) == balanced
            assert len(trace.x) == balanced

    if with_avg:
        assert len(err_plots) == len(names)
        for name, trace in zip(names, err_plots):
            assert trace.fill == "toself"
            assert trace.name == name
            assert not trace.showlegend
            if balanced:
                assert len(trace.y) == balanced * 2
                assert len(trace.x) == balanced * 2


def mock_space(x="uniform(0, 6)", y="uniform(0, 3)", **kwargs):
    """Build a mocked space"""
    mocked_config = copy.deepcopy(config)
    mocked_config["space"] = {"x": x, "y": y}
    mocked_config["space"].update(kwargs)
    return mocked_config


def mock_experiment(
    monkeypatch, ids=None, x=None, y=None, objectives=None, status=None
):
    """Mock experiment to_pandas to return given data (or default one)"""
    if ids is None:
        ids = ["a", "b", "c", "d"]
    if x is None:
        x = [0, 1, 2, 4]
    if y is None:
        y = [3, 2, 0, 1]
    if objectives is None:
        objectives = [0.1, 0.2, 0.3, 0.5]
    if status is None:
        status = ["completed", "completed", "completed", "completed"]

    def to_pandas(self, with_evc_tree=False):
        data = pandas.DataFrame(
            data={
                "id": ids,
                "x": x,
                "y": y,
                "objective": objectives,
                "status": status,
            }
        )

        return data

    monkeypatch.setattr(Experiment, "to_pandas", to_pandas)


def mock_experiment_with_random_to_pandas(monkeypatch, status=None, unbalanced=False):
    def to_pandas(self, with_evc_tree=False):
        if unbalanced:
            N = numpy.random.randint(5, 15)
        elif status is not None:
            N = len(status)
        else:
            N = 10
        ids = numpy.arange(N)
        x = numpy.random.normal(0, 0.1, size=N)
        y = numpy.random.normal(0, 0.1, size=N)
        objectives = numpy.random.normal(0, 0.1, size=N)
        if status is None:
            exp_status = ["completed"] * N
        else:
            exp_status = status

        data = pandas.DataFrame(
            data={
                "id": ids,
                "x": x,
                "y": y,
                "objective": objectives,
                "status": exp_status,
                "suggested": ids,
            }
        )

        return data

    monkeypatch.setattr(Experiment, "to_pandas", to_pandas)


def assert_lpi_plot(plot, dims):
    """Checks the layout of a LPI plot"""
    assert plot.layout.title.text == "LPI for experiment 'experiment-name'"
    assert plot.layout.xaxis.title.text == "Hyperparameters"
    assert plot.layout.yaxis.title.text == "Local Parameter Importance (LPI)"

    trace = plot.data[0]
    assert trace["x"] == tuple(dims)
    assert trace["y"][0] > trace["y"][1]


@pytest.mark.usefixtures("version_XYZ")
class TestLPI:
    """Tests the ``lpi()`` method provided by the plotly backend"""

    def test_requires_argument(self):
        """Tests that the experiment data are required."""
        with pytest.raises(ValueError):
            lpi(None)

    def test_returns_plotly_object(self):
        """Tests that the plotly backend returns a plotly object"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = lpi(experiment, random_state=1)

        assert type(plot) is plotly.graph_objects.Figure

    def test_graph_layout(self, monkeypatch):
        """Tests the layout of the plot"""
        config = mock_space()
        mock_experiment(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = lpi(experiment, random_state=1)
            # Why is this test failing?
            df = experiment.to_pandas()
            assert df["x"].tolist() == [0, 1, 2, 4]
            assert df["y"].tolist() == [3, 2, 0, 1]
            assert df["objective"].tolist() == [0.1, 0.2, 0.3, 0.5]

        assert_lpi_plot(plot, dims=["x", "y"])

    def test_experiment_worker_as_parameter(self, monkeypatch):
        """Tests that ``Experiment`` is a valid parameter"""
        config = mock_space()
        mock_experiment(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            experiment,
            _,
        ):
            plot = lpi(experiment, random_state=1)

        assert_lpi_plot(plot, dims=["x", "y"])

    def test_experiment_view_as_parameter(self, monkeypatch):
        """Tests that ``ExperimentView`` is a valid parameter"""
        config = mock_space()
        mock_experiment(monkeypatch)
        with create_experiment(config, trial_config) as (
            _,
            experiment,
            _,
        ):
            plot = lpi(ExperimentView(experiment), random_state=1)

        assert_lpi_plot(plot, dims=["x", "y"])

    def test_ignore_uncompleted_statuses(self, monkeypatch):
        """Tests that uncompleted statuses are ignored"""
        config = mock_space()
        mock_experiment(
            monkeypatch,
            ids="abcdefgh",
            x=[0, 0, 0, 1, 0, 2, 0, 3],
            y=[1, 0, 0, 2, 0, 0, 0, 3],
            objectives=[0.1, None, None, 0.2, None, 0.3, None, 0.5],
            status=[
                "completed",
                "new",
                "reserved",
                "completed",
                "broken",
                "completed",
                "interrupted",
                "completed",
            ],
        )
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = lpi(experiment)

        assert_lpi_plot(plot, dims=["x", "y"])

    def test_multidim(self, monkeypatch):
        """Tests that dimensions with shape > 1 are flattened properly"""
        config = mock_space(y="uniform(0, 3, shape=2)")
        mock_experiment(monkeypatch, y=[[3, 3], [2, 3], [1, 2], [0, 3]])
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = lpi(experiment, model_kwargs=dict(random_state=1))

        assert_lpi_plot(plot, dims=["x", "y[0]", "y[1]"])

    def test_fidelity(self, monkeypatch):
        """Tests that fidelity is supported"""
        config = mock_space(y="fidelity(1, 200, base=3)")
        mock_experiment(monkeypatch, y=[1, 3 ** 2, 1, 3 ** 4])
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = lpi(experiment, model_kwargs=dict(random_state=1))

        assert_lpi_plot(plot, dims=["x", "y"])

    def test_categorical(self, monkeypatch):
        """Tests that categorical is supported"""
        config = mock_space(y='choices(["a", "b", "c"])')
        mock_experiment(monkeypatch, y=["c", "c", "a", "b"])
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = lpi(experiment, model_kwargs=dict(random_state=1))

        assert_lpi_plot(plot, dims=["x", "y"])

    def test_categorical_multidim(self, monkeypatch):
        """Tests that multidim categorical is supported"""
        config = mock_space(y='choices(["a", "b", "c"], shape=3)')
        mock_experiment(
            monkeypatch,
            y=[["c", "b", "a"], ["c", "a", "c"], ["a", "b", "a"], ["c", "b", "b"]],
        )

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = lpi(experiment, model_kwargs=dict(random_state=1))

        assert_lpi_plot(plot, dims=["x", "y[0]", "y[1]", "y[2]"])


@pytest.mark.usefixtures("version_XYZ")
class TestRankings:
    """Tests the ``rankings()`` method provided by the plotly backend"""

    def test_requires_argument(self):
        """Tests that the experiment data are required."""
        with pytest.raises(ValueError):
            rankings(None)

    def test_returns_plotly_object(self, monkeypatch):
        """Tests that the plotly backend returns a plotly object"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = rankings([experiment, experiment])

        assert type(plot) is plotly.graph_objects.Figure

    def test_graph_layout(self, monkeypatch):
        """Tests the layout of the plot"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = rankings([experiment])

        assert_rankings_plot(plot, [f"{experiment.name}-v{experiment.version}"])

    def test_list_of_experiments(self, monkeypatch):
        """Tests the rankings with list of experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            child = orion.client.create_experiment(
                experiment.name, branching={"branch_to": "child"}
            )

            plot = rankings([experiment, child])

        # Exps are sorted alphabetically by names.
        assert_rankings_plot(
            plot, [f"{exp.name}-v{exp.version}" for exp in [child, experiment]]
        )

    def test_list_of_experiments_name_conflict(self, monkeypatch):
        """Tests the rankings with list of experiments with the same name"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            child = orion.client.create_experiment(
                experiment.name, branching={"branch_to": experiment.name}
            )
            assert child.name == experiment.name
            assert child.version == experiment.version + 1
            plot = rankings([experiment, child])

        # Exps are sorted alphabetically by names.
        assert_rankings_plot(
            plot, [f"{exp.name}-v{exp.version}" for exp in [experiment, child]]
        )

    def test_dict_of_experiments(self, monkeypatch):
        """Tests the rankings with renamed experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = rankings({"exp-1": experiment, "exp-2": experiment})

        assert_rankings_plot(plot, ["exp-1", "exp-2"])

    def test_list_of_dict_of_experiments(self, monkeypatch):
        """Tests the rankings with avg of competitions"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = rankings(
                [{"exp-1": experiment, "exp-2": experiment} for _ in range(10)]
            )

        assert_rankings_plot(plot, ["exp-1", "exp-2"], with_avg=True)

    def test_dict_of_list_of_experiments(self, monkeypatch):
        """Tests the rankings with avg of experiments separated in lists"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = rankings({"exp-1": [experiment] * 10, "exp-2": [experiment] * 10})

        assert_rankings_plot(plot, ["exp-1", "exp-2"], with_avg=True)

    def test_unbalanced_experiments(self, monkeypatch):
        """Tests the regrets with avg of unbalanced experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch, unbalanced=True)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = rankings({"exp-1": [experiment] * 10, "exp-2": [experiment] * 10})

        assert_rankings_plot(plot, ["exp-1", "exp-2"], with_avg=True, balanced=0)

    def test_ignore_uncompleted_statuses(self, monkeypatch):
        """Tests that uncompleted statuses are ignored"""
        mock_experiment_with_random_to_pandas(
            monkeypatch,
            status=[
                "completed",
                "new",
                "reserved",
                "completed",
                "broken",
                "completed",
                "interrupted",
                "completed",
            ],
        )
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = rankings([experiment])

        assert_rankings_plot(
            plot, [f"{experiment.name}-v{experiment.version}"], balanced=4
        )

    def test_unsupported_order_key(self):
        """Tests that unsupported order keys are rejected"""
        with create_experiment(config, trial_config) as (_, _, experiment):
            with pytest.raises(ValueError):
                rankings([experiment], order_by="unsupported")


@pytest.mark.usefixtures("version_XYZ")
class TestRegret:
    """Tests the ``regret()`` method provided by the plotly backend"""

    def test_requires_argument(self):
        """Tests that the experiment data are required."""
        with pytest.raises(ValueError):
            regret(None)

    def test_returns_plotly_object(self):
        """Tests that the plotly backend returns a plotly object"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = regret(experiment)

        assert type(plot) is plotly.graph_objects.Figure

    def test_graph_layout(self):
        """Tests the layout of the plot"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = regret(experiment)

        assert_regret_plot(plot)

    def test_experiment_worker_as_parameter(self):
        """Tests that ``Experiment`` is a valid parameter"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            experiment,
            _,
        ):
            plot = regret(experiment)

        assert_regret_plot(plot)

    def test_experiment_view_as_parameter(self):
        """Tests that ``ExperimentView`` is a valid parameter"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            experiment,
            _,
        ):
            plot = regret(ExperimentView(experiment))

        assert_regret_plot(plot)

    def test_ignore_uncompleted_statuses(self):
        """Tests that uncompleted statuses are ignored"""
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = regret(experiment)

        assert_regret_plot(plot)

    def test_unsupported_order_key(self):
        """Tests that unsupported order keys are rejected"""
        with create_experiment(config, trial_config) as (_, _, experiment):
            with pytest.raises(ValueError):
                regret(experiment, order_by="unsupported")


@pytest.mark.usefixtures("version_XYZ")
class TestRegrets:
    """Tests the ``regrets()`` method provided by the plotly backend"""

    def test_requires_argument(self):
        """Tests that the experiment data are required."""
        with pytest.raises(ValueError):
            regrets(None)

    def test_returns_plotly_object(self, monkeypatch):
        """Tests that the plotly backend returns a plotly object"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = regrets([experiment])

        assert type(plot) is plotly.graph_objects.Figure

    def test_graph_layout(self, monkeypatch):
        """Tests the layout of the plot"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = regrets([experiment])

        assert_regrets_plot(plot, [f"{experiment.name}-v{experiment.version}"])

    def test_list_of_experiments(self, monkeypatch):
        """Tests the regrets with list of experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            child = orion.client.create_experiment(
                experiment.name, branching={"branch_to": "child"}
            )

            plot = regrets([experiment, child])

        # Exps are sorted alphabetically by names.
        assert_regrets_plot(
            plot, [f"{exp.name}-v{exp.version}" for exp in [child, experiment]]
        )

    def test_list_of_experiments_name_conflict(self, monkeypatch):
        """Tests the regrets with list of experiments with the same name"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            child = orion.client.create_experiment(
                experiment.name, branching={"branch_to": experiment.name}
            )
            assert child.name == experiment.name
            assert child.version == experiment.version + 1
            plot = regrets([experiment, child])

        # Exps are sorted alphabetically by names.
        assert_regrets_plot(
            plot, [f"{exp.name}-v{exp.version}" for exp in [experiment, child]]
        )

    def test_dict_of_experiments(self, monkeypatch):
        """Tests the regrets with renamed experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = regrets({"exp-1": experiment, "exp-2": experiment})

        assert_regrets_plot(plot, ["exp-1", "exp-2"])

    def test_dict_of_list_of_experiments(self, monkeypatch):
        """Tests the regrets with avg of experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = regrets({"exp-1": [experiment] * 10, "exp-2": [experiment] * 10})

        assert_regrets_plot(plot, ["exp-1", "exp-2"], with_avg=True)

    def test_unbalanced_experiments(self, monkeypatch):
        """Tests the regrets with avg of unbalanced experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch, unbalanced=True)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = regrets({"exp-1": [experiment] * 10, "exp-2": [experiment] * 10})

        assert_regrets_plot(plot, ["exp-1", "exp-2"], with_avg=True, balanced=0)

    def test_ignore_uncompleted_statuses(self, monkeypatch):
        """Tests that uncompleted statuses are ignored"""
        mock_experiment_with_random_to_pandas(
            monkeypatch,
            status=[
                "completed",
                "new",
                "reserved",
                "completed",
                "broken",
                "completed",
                "interrupted",
                "completed",
            ],
        )
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = regrets([experiment])

        assert_regrets_plot(
            plot, [f"{experiment.name}-v{experiment.version}"], balanced=4
        )

    def test_unsupported_order_key(self):
        """Tests that unsupported order keys are rejected"""
        with create_experiment(config, trial_config) as (_, _, experiment):
            with pytest.raises(ValueError):
                regrets([experiment], order_by="unsupported")


def assert_parallel_coordinates_plot(plot, order):
    """Checks the layout of a parallel coordinates plot"""
    assert (
        plot.layout.title.text
        == "Parallel Coordinates Plot for experiment 'experiment-name'"
    )

    trace = plot.data[0]
    for i in range(len(order)):
        assert trace.dimensions[i].label == order[i]


@pytest.mark.usefixtures("version_XYZ")
class TestParallelCoordinates:
    """Tests the ``parallel_coordinates()`` method provided by the plotly backend"""

    def test_requires_argument(self):
        """Tests that the experiment data are required."""
        with pytest.raises(ValueError):
            parallel_coordinates(None)

    def test_returns_plotly_object(self):
        """Tests that the plotly backend returns a plotly object"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = parallel_coordinates(experiment)

        assert type(plot) is plotly.graph_objects.Figure

    def test_graph_layout(self):
        """Tests the layout of the plot"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = parallel_coordinates(experiment)

        assert_parallel_coordinates_plot(plot, order=["x", "loss"])

    def test_experiment_worker_as_parameter(self):
        """Tests that ``Experiment`` is a valid parameter"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            experiment,
            _,
        ):
            plot = parallel_coordinates(experiment)

        assert_parallel_coordinates_plot(plot, order=["x", "loss"])

    def test_experiment_view_as_parameter(self):
        """Tests that ``ExperimentView`` is a valid parameter"""
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            experiment,
            _,
        ):
            plot = parallel_coordinates(ExperimentView(experiment))

        assert_parallel_coordinates_plot(plot, order=["x", "loss"])

    def test_ignore_uncompleted_statuses(self):
        """Tests that uncompleted statuses are ignored"""
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = parallel_coordinates(experiment)

        assert_parallel_coordinates_plot(plot, order=["x", "loss"])

    def test_unsupported_order_key(self):
        """Tests that unsupported order keys are rejected"""
        with create_experiment(config, trial_config) as (_, _, experiment):
            with pytest.raises(ValueError):
                parallel_coordinates(experiment, order=["unsupported"])

    def test_order_columns(self):
        """Tests that columns are sorted according to ``order``"""
        multidim_config = copy.deepcopy(config)
        for k in "yzutv":
            multidim_config["space"][k] = "uniform(0, 200)"
        with create_experiment(multidim_config, trial_config) as (_, _, experiment):
            plot = parallel_coordinates(experiment, order="vzyx")

        assert_parallel_coordinates_plot(plot, order=["v", "z", "y", "x", "loss"])

    def test_multidim(self):
        """Tests that dimensions with shape > 1 are flattened properly"""
        multidim_config = copy.deepcopy(config)
        multidim_config["space"]["y"] = "uniform(0, 200, shape=4)"
        with create_experiment(multidim_config, trial_config) as (_, _, experiment):
            plot = parallel_coordinates(experiment)

        assert_parallel_coordinates_plot(
            plot, order=["x", "y[0]", "y[1]", "y[2]", "y[3]", "loss"]
        )

    def test_fidelity(self):
        """Tests that fidelity is set to first column by default"""
        fidelity_config = copy.deepcopy(config)
        fidelity_config["space"]["z"] = "fidelity(1, 200, base=3)"
        with create_experiment(fidelity_config, trial_config) as (_, _, experiment):
            plot = parallel_coordinates(experiment)

        assert_parallel_coordinates_plot(plot, order=["z", "x", "loss"])

    def test_categorical(self):
        """Tests that categorical is supported"""
        categorical_config = copy.deepcopy(config)
        categorical_config["space"]["z"] = 'choices(["a", "b", "c"])'
        with create_experiment(categorical_config, trial_config) as (_, _, experiment):
            plot = parallel_coordinates(experiment)

        assert_parallel_coordinates_plot(plot, order=["x", "z", "loss"])

    def test_categorical_multidim(self):
        """Tests that multidim categorical is supported"""
        categorical_config = copy.deepcopy(config)
        categorical_config["space"]["z"] = 'choices(["a", "b", "c"], shape=3)'
        with create_experiment(categorical_config, trial_config) as (_, _, experiment):
            plot = parallel_coordinates(experiment)

        assert_parallel_coordinates_plot(
            plot, order=["x", "z[0]", "z[1]", "z[2]", "loss"]
        )
