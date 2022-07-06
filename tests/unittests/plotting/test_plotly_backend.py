"""Collection of tests for :mod:`orion.plotting.backend_plotly`."""
import copy
import datetime

import numpy
import pandas
import plotly
import pytest

import orion.client
from orion.analysis.partial_dependency_utils import partial_dependency_grid
from orion.core.worker.experiment import Experiment
from orion.plotting.backend_plotly import infer_unit_time
from orion.plotting.base import (
    durations,
    lpi,
    parallel_assessment,
    parallel_coordinates,
    partial_dependencies,
    rankings,
    regret,
    regrets,
)
from orion.testing import create_experiment
from orion.testing.plotting import (
    assert_durations_plot,
    assert_lpi_plot,
    assert_parallel_coordinates_plot,
    assert_partial_dependencies_plot,
    assert_rankings_plot,
    assert_regret_plot,
    assert_regrets_plot,
    asset_parallel_assessment_plot,
)

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


def mock_space(x="uniform(0, 6)", y="uniform(0, 3)", **kwargs):
    """Build a mocked space"""
    mocked_config = copy.deepcopy(config)
    mocked_config["space"] = {"x": x}
    if y is not None:
        mocked_config["space"]["y"] = y
    mocked_config["space"].update(kwargs)
    return mocked_config


def mock_experiment(
    monkeypatch, ids=None, x=None, y=None, z=None, objectives=None, status=None
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

    data = {
        "id": ids,
        "x": x,
        "objective": objectives,
        "status": status,
        "suggested": ids,
    }

    if not isinstance(y, str):
        data["y"] = y
    if z is not None:
        data["z"] = z

    def to_pandas(self, with_evc_tree=False):

        return pandas.DataFrame(data=data)

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
        start = datetime.datetime.utcnow()
        suggested = [start + datetime.timedelta(seconds=int(i)) for i in ids]
        if status is None:
            exp_status = ["completed"] * N
        else:
            exp_status = status

        completed = [start + datetime.timedelta(seconds=int(i) + 1) for i in ids]

        data = pandas.DataFrame(
            data={
                "id": ids,
                "x": x,
                "y": y,
                "objective": objectives,
                "status": exp_status,
                "suggested": suggested,
                "completed": completed,
            }
        )

        return data

    monkeypatch.setattr(Experiment, "to_pandas", to_pandas)


def mock_model():
    """Return a mocked regressor which just predict iterated integers"""

    class Model:
        """Mocked Regressor"""

        def __init__(self):
            self.i = 0

        def predict(self, data):
            """Returns counting of predictions requested."""
            data = numpy.arange(data.shape[0]) + self.i
            self.i += data.shape[0]
            return data  # + numpy.random.normal(0, self.i, size=data.shape[0])

    return Model()


def mock_train_regressor(monkeypatch, assert_model=None, assert_model_kwargs=None):
    """Mock the train_regressor to return the mocked regressor instead"""

    def train_regressor(model, data, **kwargs):
        """Return the mocked model, and then model argument if requested"""
        if assert_model:
            assert model == assert_model
        if assert_model_kwargs:
            assert kwargs == assert_model_kwargs
        return mock_model()

    monkeypatch.setattr(
        "orion.analysis.partial_dependency_utils.train_regressor", train_regressor
    )


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
            plot = lpi(experiment, model_kwargs=dict(random_state=1))

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
            plot = lpi(experiment, model_kwargs=dict(random_state=1))
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
            plot = lpi(experiment, model_kwargs=dict(random_state=1))

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
        mock_experiment(monkeypatch, y=[1, 3**2, 1, 3**4])
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


@pytest.mark.usefixtures("version_XYZ")
class TestPartialDependencies:
    """Tests the ``partial_dependencies()`` method provided by the plotly backend"""

    def test_returns_plotly_object(self, monkeypatch):
        """Tests that the plotly backend returns a plotly object"""
        mock_train_regressor(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert type(plot) is plotly.graph_objects.Figure

    def test_graph_layout(self, monkeypatch):
        """Tests the layout of the plot"""
        mock_train_regressor(monkeypatch)
        config = mock_space()
        mock_experiment(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )
            df = experiment.to_pandas()
            assert df["x"].tolist() == [0, 1, 2, 4]
            assert df["y"].tolist() == [3, 2, 0, 1]
            assert df["objective"].tolist() == [0.1, 0.2, 0.3, 0.5]

        assert_partial_dependencies_plot(plot, dims=["x", "y"])

    def test_ignore_uncompleted_statuses(self, monkeypatch):
        """Tests that uncompleted statuses are ignored"""
        mock_train_regressor(monkeypatch)
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
            plot = partial_dependencies(experiment, n_grid_points=5)

        assert_partial_dependencies_plot(plot, dims=["x", "y"])

    def test_multidim(self, monkeypatch):
        """Tests that dimensions with shape > 1 are flattened properly"""
        mock_train_regressor(monkeypatch)
        config = mock_space(y="uniform(0, 3, shape=2)")
        mock_experiment(monkeypatch, y=[[3, 3], [2, 3], [1, 2], [0, 3]])
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert_partial_dependencies_plot(plot, dims=["x", "y[0]", "y[1]"])

    def test_fidelity(self, monkeypatch):
        """Tests that fidelity is supported"""
        mock_train_regressor(monkeypatch)
        config = mock_space(y="fidelity(1, 200, base=3)")
        mock_experiment(monkeypatch, y=[1, 3**2, 1, 3**4])
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert_partial_dependencies_plot(plot, dims=["x", "y"], log_dims=["y"])

    def test_categorical(self, monkeypatch):
        """Tests that categorical is supported"""
        mock_train_regressor(monkeypatch)
        config = mock_space(y='choices(["a", "b", "c"])')
        mock_experiment(monkeypatch, y=["c", "c", "a", "b"])
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        # There is only 3 categories, so test must be adjusted accordingly.
        assert_partial_dependencies_plot(
            plot, dims=["x", "y"], n_grid_points={"x": 5, "y": 3}
        )

    def test_categorical_multidim(self, monkeypatch):
        """Tests that multidim categorical is supported"""
        mock_train_regressor(monkeypatch)
        config = mock_space(y='choices(["a", "b", "c"], shape=3)')
        mock_experiment(
            monkeypatch,
            y=[["c", "b", "a"], ["c", "a", "c"], ["a", "b", "a"], ["c", "b", "b"]],
        )

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert_partial_dependencies_plot(
            plot,
            dims=["x", "y[0]", "y[1]", "y[2]"],
            n_grid_points={"x": 5, "y[0]": 3, "y[1]": 3, "y[2]": 3},
        )

    def test_logarithmic_scales_first(self, monkeypatch):
        """Test that log dims are turn to log scale

        Test first dim specifically because special xaxis name for first dim.
        """
        mock_train_regressor(monkeypatch)
        config = mock_space(x="loguniform(0.001, 1)", z="uniform(0, 1)")
        mock_experiment(monkeypatch, x=[0.001, 0.1, 0.01, 1], z=[0, 0.1, 0.2, 0.5])

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert_partial_dependencies_plot(
            plot, dims=["x", "y", "z"], n_grid_points=5, log_dims=["x"]
        )

    def test_logarithmic_scales_any_dim(self, monkeypatch):
        """Test that log dims are turn to log scale"""
        mock_train_regressor(monkeypatch)
        config = mock_space(y="loguniform(0.001, 1)", z="uniform(0, 1)")
        mock_experiment(monkeypatch, y=[0.001, 0.1, 0.01, 1], z=[0, 0.1, 0.2, 0.5])

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert_partial_dependencies_plot(
            plot, dims=["x", "y", "z"], n_grid_points=5, log_dims=["y"]
        )

    def test_int_logarithmic_scales(self, monkeypatch):
        """Test that int log dims are turn to log scale"""
        mock_train_regressor(monkeypatch)
        config = mock_space(y="loguniform(1, 1000, discrete=True)", z="uniform(0, 1)")
        mock_experiment(monkeypatch, y=[1, 10, 100, 1000], z=[0, 0.1, 0.2, 0.5])

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert_partial_dependencies_plot(
            plot, dims=["x", "y", "z"], n_grid_points=5, log_dims=["y"]
        )

    def test_one_param(self, monkeypatch):
        """Test plotting a space with only 1 dim"""
        mock_train_regressor(monkeypatch)
        config = mock_space(y=None)
        mock_experiment(monkeypatch, y="drop")

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert_partial_dependencies_plot(plot, dims=["x"], n_grid_points=5)

    def test_select_params(self, monkeypatch):
        """Test selecting subset"""
        mock_train_regressor(monkeypatch)
        config = mock_space(z="uniform(0, 1)")
        mock_experiment(monkeypatch, z=[0, 0.1, 0.2, 0.5])

        for params in [["x"], ["x", "y"], ["y", "z"]]:
            with create_experiment(config, trial_config) as (_, _, experiment):
                plot = partial_dependencies(
                    experiment,
                    params=params,
                    n_grid_points=5,
                    model_kwargs=dict(random_state=1),
                )

            assert_partial_dependencies_plot(plot, dims=params, n_grid_points=5)

    def test_custom_smoothing(self, monkeypatch):
        """Test changing smoothing value"""
        mock_train_regressor(monkeypatch)
        config = mock_space()
        mock_experiment(monkeypatch)

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment,
                n_grid_points=5,
                model_kwargs=dict(random_state=1),
                smoothing=1.2,
            )

        with pytest.raises(AssertionError):
            assert_partial_dependencies_plot(plot, dims=["x", "y"], n_grid_points=5)

        assert_partial_dependencies_plot(
            plot, dims=["x", "y"], n_grid_points=5, smoothing=1.2
        )

    def test_custom_n_grid_points(self, monkeypatch):
        """Test changing n_grid_points value"""
        mock_train_regressor(monkeypatch)
        config = mock_space()
        mock_experiment(monkeypatch)

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment,
                n_grid_points=10,
                model_kwargs=dict(random_state=1),
            )

        with pytest.raises(AssertionError):
            assert_partial_dependencies_plot(plot, dims=["x", "y"], n_grid_points=5)

        assert_partial_dependencies_plot(plot, dims=["x", "y"], n_grid_points=10)

    def test_custom_n_samples(self, monkeypatch):
        """Test changing n_samples value"""
        mock_train_regressor(monkeypatch)
        config = mock_space()
        mock_experiment(monkeypatch)

        PARAMS = ["x", "y"]
        N_SAMPLES = numpy.random.randint(20, 50)

        def mock_partial_dependency_grid(space, model, params, samples, n_points):
            print(samples)
            assert samples.shape == (N_SAMPLES, len(PARAMS))
            return partial_dependency_grid(space, model, params, samples, n_points)

        monkeypatch.setattr(
            "orion.analysis.partial_dependency_utils.partial_dependency_grid",
            mock_partial_dependency_grid,
        )

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment,
                n_grid_points=10,
                model_kwargs=dict(random_state=1),
                n_samples=N_SAMPLES,
            )

    def test_custom_colorscale(self, monkeypatch):
        """Test changing colorscale"""
        mock_train_regressor(monkeypatch)
        config = mock_space()
        mock_experiment(monkeypatch)

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment,
                n_grid_points=5,
                colorscale="Viridis",
                model_kwargs=dict(random_state=1),
            )

        with pytest.raises(AssertionError):
            assert_partial_dependencies_plot(
                plot, dims=["x", "y"], n_grid_points=5, custom_colorscale=False
            )

        assert_partial_dependencies_plot(
            plot, dims=["x", "y"], n_grid_points=5, custom_colorscale=True
        )

    def test_custom_model(self, monkeypatch):
        """Test changing type of regression model"""
        mock_train_regressor(monkeypatch, assert_model="BaggingRegressor")
        config = mock_space()
        mock_experiment(monkeypatch)

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment,
                n_grid_points=5,
                model="BaggingRegressor",
                model_kwargs=dict(random_state=1),
            )

    def test_custom_model_kwargs(self, monkeypatch):
        """Test changing arguments of regression model"""
        mock_train_regressor(monkeypatch, assert_model_kwargs=dict(random_state=1))
        config = mock_space()
        mock_experiment(monkeypatch)

        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment,
                n_grid_points=5,
                model_kwargs=dict(random_state=1),
            )


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
                experiment.name, branching={"branch_to": "child", "enable": True}
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
                experiment.name,
                branching={"branch_to": experiment.name, "enable": True},
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
                experiment.name, branching={"branch_to": "child", "enable": True}
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
                experiment.name,
                branching={"branch_to": experiment.name, "enable": True},
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


@pytest.mark.usefixtures("version_XYZ")
class TestParallelAdvantage:
    """Tests the ``parallel_assessment()`` method provided by the plotly backend"""

    def test_requires_argument(self):
        """Tests that the experiment data are required."""
        with pytest.raises(ValueError):
            parallel_assessment(None)

    def test_returns_plotly_object(self, monkeypatch):
        """Tests that the plotly backend returns a plotly object"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = parallel_assessment({"random": [experiment]})

        assert type(plot) is plotly.graph_objects.Figure

    def test_graph_layout(self, monkeypatch):
        """Tests the layout of the plot"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = parallel_assessment({"random": [experiment] * 2})

        asset_parallel_assessment_plot(plot, ["random"], 1)

    def test_list_of_experiments(self, monkeypatch):
        """Tests the parallel_assessment with list of experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            child = orion.client.create_experiment(
                experiment.name, branching={"branch_to": "child", "enable": True}
            )

            plot = parallel_assessment({"random": [experiment, child]})

        asset_parallel_assessment_plot(plot, ["random"], 1)

        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = parallel_assessment(
                {"exp-1": [experiment] * 10, "exp-2": [experiment] * 10}
            )

        asset_parallel_assessment_plot(plot, ["exp-1", "exp-2"], 1)

    def test_list_of_experiments_name_conflict(self, monkeypatch):
        """Tests the parallel_assessment with list of experiments with the same name"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            child = orion.client.create_experiment(
                experiment.name,
                branching={"branch_to": experiment.name, "enable": True},
            )
            assert child.name == experiment.name
            assert child.version == experiment.version + 1
            plot = parallel_assessment({"random": [experiment, child]})

        asset_parallel_assessment_plot(plot, ["random"], 1)

    def test_dict_of_experiments(self, monkeypatch):
        """Tests the parallel_assessment with renamed experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = parallel_assessment({"exp-1": experiment, "exp-2": experiment})

        asset_parallel_assessment_plot(plot, ["exp-1", "exp-2"], 1)

    def test_dict_of_list_of_experiments(self, monkeypatch):
        """Tests the regrparallel_assessmentets with avg of experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = parallel_assessment(
                {"exp-1": [experiment] * 10, "exp-2": [experiment] * 10}
            )

        asset_parallel_assessment_plot(plot, ["exp-1", "exp-2"], 1)

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
            plot = parallel_assessment({"random": [experiment]})

        asset_parallel_assessment_plot(plot, ["random"], 1)


@pytest.mark.usefixtures("version_XYZ")
class TestDurations:
    """Tests the ``durations()`` method provided by the plotly backend"""

    def test_requires_argument(self):
        """Tests that the experiment data are required."""
        with pytest.raises(ValueError):
            durations(None)

    def test_returns_plotly_object(self, monkeypatch):
        """Tests that the plotly backend returns a plotly object"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = durations([experiment])

        assert type(plot) is plotly.graph_objects.Figure

    def test_graph_layout(self, monkeypatch):
        """Tests the layout of the plot"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = durations([experiment])

        assert_durations_plot(plot, [f"{experiment.name}-v{experiment.version}"])

    def test_list_of_experiments(self, monkeypatch):
        """Tests the regrets with list of experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            child = orion.client.create_experiment(
                experiment.name, branching={"branch_to": "child", "enable": True}
            )

            plot = durations([experiment, child])

        # Exps are sorted alphabetically by names.
        assert_durations_plot(
            plot, [f"{exp.name}-v{exp.version}" for exp in [child, experiment]]
        )

    def test_list_of_experiments_name_conflict(self, monkeypatch):
        """Tests the durations with list of experiments with the same name"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            child = orion.client.create_experiment(
                experiment.name,
                branching={"branch_to": experiment.name, "enable": True},
            )
            assert child.name == experiment.name
            assert child.version == experiment.version + 1
            plot = durations([experiment, child])

        # Exps are sorted alphabetically by names.
        assert_durations_plot(
            plot, [f"{exp.name}-v{exp.version}" for exp in [experiment, child]]
        )

    def test_dict_of_experiments(self, monkeypatch):
        """Tests the durations with renamed experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = durations({"exp-1": experiment, "exp-2": experiment})

        assert_durations_plot(plot, ["exp-1", "exp-2"])

    def test_dict_of_list_of_experiments(self, monkeypatch):
        """Tests the regrets with avg of experiments"""
        mock_experiment_with_random_to_pandas(monkeypatch)
        with create_experiment(config, trial_config, ["completed"]) as (
            _,
            _,
            experiment,
        ):
            plot = durations({"exp-1": [experiment] * 10, "exp-2": [experiment] * 10})

        assert_durations_plot(plot, ["exp-1", "exp-2"])

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
            plot = durations([experiment])

        assert_durations_plot(plot, [f"{experiment.name}-v{experiment.version}"])


@pytest.mark.usefixtures("version_XYZ")
class TestDurationUnitTime:
    """Test `infer_unit_time`"""

    @pytest.mark.parametrize(
        "time_unit, duration",
        (
            ["day(s)", [60 * 60 * 24, 60 * 60 * 24 * 2, 60 * 60 * 24 * 3]],
            ["hour(s)", [60 * 60, 60 * 60 * 2, 60 * 60 * 3]],
            ["minute(s)", [60, 60 * 2, 60 * 3]],
            ["second(s)", [1, 2, 3]],
        ),
    )
    def test_duration_units(self, time_unit, duration):
        """Test that correct time unit and duration will be return with different duration values in seconds"""
        df = pandas.DataFrame({"duration": duration})

        duration, unit = infer_unit_time(df, min_unit=1)
        assert unit == time_unit
        assert duration.tolist() == [1, 2, 3]

        duration, unit = infer_unit_time(df, min_unit=3)
        if time_unit != "second(s)":
            assert unit != time_unit
        else:
            assert unit == time_unit

        df["duration"] = df["duration"] * 3
        duration, unit = infer_unit_time(df, min_unit=3)
        assert unit == time_unit
        assert duration.tolist() == [3, 6, 9]

    @pytest.mark.parametrize(
        "time_unit, duration",
        (
            ["day(s)", [60 * 60 * 24 * 3, 60 * 60 * 24 * 2, 60 * 60 * 24]],
            ["hour(s)", [60 * 60 * 3, 60 * 60 * 2, 60 * 60]],
            ["minute(s)", [60 * 3, 60 * 2, 60]],
            ["second(s)", [3, 2, 1]],
        ),
    )
    def test_parallel_assessment_duration_units(self, time_unit, duration):
        """Test that correct time unit and duration will be return with different duration values in seconds"""
        df = pandas.DataFrame({"duration": duration})

        duration, unit = infer_unit_time(df, min_unit=1)
        assert unit == time_unit
        assert duration.tolist() == [3, 2, 1]

        duration, unit = infer_unit_time(df, min_unit=3)
        if time_unit != "second(s)":
            assert unit != time_unit
        else:
            assert unit == time_unit

        df["duration"] = df["duration"] * 3
        duration, unit = infer_unit_time(df, min_unit=3)
        assert unit == time_unit
        assert duration.tolist() == [9, 6, 3]
