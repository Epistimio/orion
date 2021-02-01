"""Collection of tests for :mod:`orion.plotting.backend_plotly`."""
import copy

import numpy
import pandas
import plotly
import pytest

from orion.analysis.partial_dependency_utils import partial_dependency_grid
from orion.core.worker.experiment import Experiment
from orion.plotting.base import lpi, parallel_coordinates, partial_dependencies, regret
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
    }

    if not isinstance(y, str):
        data["y"] = y
    if z is not None:
        data["z"] = z

    def to_pandas(self, with_evc_tree=False):

        return pandas.DataFrame(data=data)

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
            return data  #  + numpy.random.normal(0, self.i, size=data.shape[0])

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


def assert_lpi_plot(plot, dims):
    """Checks the layout of a LPI plot"""
    assert plot.layout.title.text == "LPI for experiment 'experiment-name'"
    assert plot.layout.xaxis.title.text == "Hyperparameters"
    assert plot.layout.yaxis.title.text == "Local Parameter Importance (LPI)"

    trace = plot.data[0]
    assert trace["x"] == tuple(dims)
    assert trace["y"][0] > trace["y"][1]


def assert_partial_dependencies_plot(
    plot,
    dims,
    custom_colorscale=False,
    smoothing=0.85,
    n_grid_points=5,
    n_samples=4,
    log_dims=None,
):
    """Checks the layout of a partial dependencies plot"""
    if not isinstance(n_grid_points, dict):
        n_grid_points = {dim: n_grid_points for dim in dims}
    if log_dims is None:
        log_dims = {}

    def _ax_label(axis, index):
        if index == 0:
            return f"{axis}axis"

        return f"{axis}axis{index + 1}"

    def _ax_layout(axis, index):
        return plot.layout[_ax_label(axis, index)]

    assert (
        plot.layout.title.text
        == "Partial dependencies for experiment 'experiment-name'"
    )

    assert plot.layout.coloraxis.colorbar.title.text == "Objective"
    assert plot.layout.yaxis.title.text == "Objective"

    yrange = _ax_layout("y", 0).range

    def all_indices():
        return {
            j * len(dims) + i + 1 for i in range(len(dims)) for j in range(i, len(dims))
        }

    def first_column():
        return {i * len(dims) + 1 for i in range(len(dims))}

    def last_row():
        return {len(dims) * (len(dims) - 1) + i + 1 for i in range(len(dims))}

    def diagonal():
        return {i * len(dims) + i + 1 for i in range(len(dims))}

    def assert_axis_log(axis, index, name):
        axis_type = _ax_layout(axis, index).type
        if name in log_dims:
            assert axis_type == "log"
        else:
            assert axis_type != "log"

    def assert_log_x():
        x_tested = set()
        for dim_i, dim_name in enumerate(dims):
            x_index = dim_i * len(dims) + dim_i
            for row in range(dim_i, len(dims)):
                assert_axis_log("x", x_index, dim_name)
                x_tested.add(x_index + 1)
                x_index += len(dims)

        assert x_tested == all_indices()

    assert_log_x()

    def assert_shared_y_on_diagonal():
        y_tested = set()
        for dim_i, dim_name in enumerate(dims):
            # Test shared y axis across the diagonal
            y_index = dim_i * len(dims) + dim_i
            assert _ax_layout("y", y_index).range == yrange
            y_tested.add(y_index + 1)

        assert y_tested == diagonal()

    assert_shared_y_on_diagonal()

    def assert_log_y():
        y_tested = set()
        for dim_i, dim_name in enumerate(dims):
            # Test shared y axis across the diagonal
            y_index = dim_i * len(dims) + dim_i
            # Should not be log
            assert_axis_log("y", y_index, None)
            y_tested.add(y_index + 1)

            y_index = dim_i * len(dims)
            for column in range(max(dim_i, 0)):
                assert_axis_log("y", y_index, dim_name)
                y_tested.add(y_index + 1)
                y_index += 1

        assert y_tested == all_indices()

    assert_log_y()

    def assert_x_labels():
        x_tested = set()
        for dim_i, dim_name in enumerate(dims):
            x_index = len(dims) * (len(dims) - 1) + dim_i
            assert _ax_layout("x", x_index).title.text == dim_name

        assert x_tested == last_row()

    def assert_y_labels():
        y_tested = set()
        for dim_i, dim_name in enumerate(dims):
            if dim_i > 0:
                # Test lable at left of row
                y_index = dim_i * len(dims)
                assert _ax_layout("y", y_index).title.text == dim_name
                y_tested.add(y_index + 1)
            else:
                assert _ax_layout("y", 0).title.text == "Objective"
                y_tested.add(1)

        assert y_tested == first_column()

    assert_y_labels()

    # assert x_tested == {1, 4, 5, 7, 8, 9}
    # assert y_tested == {1, 4, 5, 7, 8, 9}

    if custom_colorscale:
        assert plot.layout.coloraxis.colorscale[0][1] != "rgb(247,251,255)"
    else:
        assert plot.layout.coloraxis.colorscale[0][1] == "rgb(247,251,255)"

    data = plot.data
    data_index = 0
    for x_i, x_name in enumerate(dims):

        # Test scatter mean
        assert data[data_index].mode == "lines"
        assert data[data_index].showlegend is False
        assert len(data[data_index].x) == n_grid_points[x_name]
        assert len(data[data_index].y) == n_grid_points[x_name]
        data_index += 1
        # Test scatter var
        assert data[data_index].mode == "lines"
        assert data[data_index].fill == "toself"
        assert data[data_index].showlegend is False
        assert len(data[data_index].x) == 2 * n_grid_points[x_name]
        assert len(data[data_index].y) == 2 * n_grid_points[x_name]
        data_index += 1

        for y_i in range(x_i + 1, len(dims)):
            y_name = dims[y_i]

            # Test contour
            assert data[data_index].line.smoothing == smoothing
            # To share colorscale across subplots
            assert data[data_index].coloraxis == "coloraxis"
            assert len(data[data_index].x) == n_grid_points[x_name]
            assert len(data[data_index].y) == n_grid_points[y_name]
            assert data[data_index].z.shape == (
                n_grid_points[y_name],
                n_grid_points[x_name],
            )
            data_index += 1

            # Test scatter
            assert data[data_index].mode == "markers"
            assert data[data_index].showlegend is False
            assert len(data[data_index].x) == n_samples
            assert len(data[data_index].y) == n_samples
            data_index += 1

    # Make sure we covered all data
    assert len(data) == data_index


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
        mock_experiment(monkeypatch, y=[1, 3 ** 2, 1, 3 ** 4])
        with create_experiment(config, trial_config) as (_, _, experiment):
            plot = partial_dependencies(
                experiment, n_grid_points=5, model_kwargs=dict(random_state=1)
            )

        assert_partial_dependencies_plot(plot, dims=["x", "y"])

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
        """Test ploting a space with only 1 dim"""
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
