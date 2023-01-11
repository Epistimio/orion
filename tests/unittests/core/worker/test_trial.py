#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.worker.trial`."""
import copy
import datetime
import os
import warnings

import bson
import numpy
import pytest

from orion.core.worker.trial import Trial


@pytest.fixture
def base_trial():
    x = {"name": "/x", "value": [1, 2], "type": "real"}
    y = {"name": "/y", "value": [1, 2], "type": "integer"}
    objective = {"name": "objective", "value": 10, "type": "objective"}

    return Trial(
        experiment=1,
        status="completed",
        params=[x, y],
        results=[objective],
        exp_working_dir="/some/path",
    )


@pytest.fixture
def params():
    return [
        dict(
            name="/decoding_layer",
            type="categorical",
            value="lstm_with_attention",
        ),
        dict(name="/encoding_layer", type="categorical", value="gru"),
    ]


@pytest.fixture
def trial_config(params):
    return dict(
        _id="something-unique",
        id="eebf174e46dae012521c12985b652cff",
        experiment="supernaedo2-dendi",
        exp_working_dir=None,
        status="completed",
        worker="23415151",
        submit_time="2017-11-22T23:00:00",
        start_time=150,
        end_time="2017-11-23T00:00:00",
        heartbeat=None,
        results=[
            dict(
                name="objective-name",
                type="objective",
                value=2,
            ),
            dict(
                name="gradient-name",
                type="gradient",
                value=[-0.1, 2],
            ),
        ],
        params=params,
        parent=None,
    )


@pytest.fixture
def trial_config_with_correct_start_time(params):
    return dict(
        _id="something-unique",
        id="eebf174e46dae012521c12985b652cff",
        experiment="supernaedo2-dendi",
        exp_working_dir=None,
        status="completed",
        worker="23415151",
        submit_time=datetime.datetime.fromisoformat("2017-11-22T23:00:00"),
        start_time=datetime.datetime.fromisoformat("2017-11-22T23:00:00"),
        end_time=datetime.datetime.fromisoformat("2017-11-23T00:00:00"),
        heartbeat=None,
        results=[
            dict(
                name="objective-name",
                type="objective",
                value=2,
            ),
            dict(
                name="gradient-name",
                type="gradient",
                value=[-0.1, 2],
            ),
        ],
        params=params,
        parent=None,
    )


@pytest.fixture
def trial_config_no_end_time(params):
    return dict(
        _id="something-unique",
        id="eebf174e46dae012521c12985b652cff",
        experiment="supernaedo2-dendi",
        exp_working_dir=None,
        status="completed",
        worker="23415151",
        submit_time=datetime.datetime.fromisoformat("2017-11-22T23:00:00"),
        start_time=datetime.datetime.fromisoformat("2017-11-22T23:00:00"),
        end_time=None,
        heartbeat=datetime.datetime.fromisoformat("2017-11-22T00:30:00"),
        results=[
            dict(
                name="objective-name",
                type="objective",
                value=2,
            ),
            dict(
                name="gradient-name",
                type="gradient",
                value=[-0.1, 2],
            ),
        ],
        params=params,
        parent=None,
    )


class TestTrial:
    """Test Trial object and class."""

    def test_init_empty(self):
        """Initialize empty trial."""
        t = Trial()
        assert t.experiment is None
        assert t.status == "new"
        assert t.worker is None
        assert t.submit_time is None
        assert t.start_time is None
        assert t.end_time is None
        assert t.results == []
        assert t.params == {}
        assert t.exp_working_dir is None

    def test_init_full(self, trial_config):
        """Initialize with a dictionary with complete specification."""
        t = Trial(**trial_config)
        assert t.experiment == trial_config["experiment"]
        assert t.status == trial_config["status"]
        assert t.worker == trial_config["worker"]
        assert t.submit_time == trial_config["submit_time"]
        assert t.start_time == trial_config["start_time"]
        assert t.end_time == trial_config["end_time"]
        assert list(map(lambda x: x.to_dict(), t.results)) == trial_config["results"]
        assert t.results[0].name == trial_config["results"][0]["name"]
        assert t.results[0].type == trial_config["results"][0]["type"]
        assert t.results[0].value == trial_config["results"][0]["value"]
        assert list(map(lambda x: x.to_dict(), t._params)) == trial_config["params"]
        assert t.exp_working_dir is None

    def test_higher_shapes_not_ndarray(self):
        """Test that `numpy.ndarray` values are converted to list."""
        value = numpy.zeros([3])
        expected = value.tolist()
        params = [dict(name="/x", type="real", value=value)]
        trial = Trial(params=params)

        assert trial._params[0].value == expected

    def test_bad_access(self):
        """Other than `Trial.__slots__` are not allowed."""
        t = Trial()
        with pytest.raises(AttributeError):
            t.asdfa = 3

    def test_bad_init(self):
        """Other than `Trial.__slots__` are not allowed in __init__ too."""
        with pytest.raises(AttributeError):
            Trial(ispii="iela")

    def test_value_bad_init(self):
        """Other than `Trial.Value.__slots__` are not allowed in __init__ too."""
        with pytest.raises(AttributeError):
            Trial.Value(ispii="iela")

    def test_param_bad_init(self):
        """Other than `Trial.Param.__slots__` are not allowed in __init__ too."""
        with pytest.raises(AttributeError):
            Trial.Param(ispii="iela")

    def test_result_bad_init(self):
        """Other than `Trial.Result.__slots__` are not allowed in __init__ too."""
        with pytest.raises(AttributeError):
            Trial.Result(ispii="iela")

    def test_not_allowed_status(self):
        """Other than `Trial.allowed_stati` are not allowed in `Trial.status`."""
        t = Trial()
        with pytest.raises(ValueError):
            t.status = "asdf"
        with pytest.raises(ValueError):
            t = Trial(status="ispi")

    def test_value_not_allowed_type(self):
        """Other than `Trial.Result.allowed_types` are not allowed in `Trial.Result.type`.

        Same for `Trial.Param`.
        """
        with pytest.raises(ValueError):
            v = Trial.Result(name="asfda", type="hoho")
        v = Trial.Result()
        with pytest.raises(ValueError):
            v.type = "asfda"

        with pytest.raises(ValueError):
            v = Trial.Param(name="asfda", type="hoho")
        v = Trial.Param()
        with pytest.raises(ValueError):
            v.type = "asfda"

    def test_conversion_to_dict(self, trial_config):
        """Convert to dictionary form for database using ``dict``."""
        t = Trial(**trial_config)
        assert t.to_dict() == trial_config

    def test_build_trials(self, exp_config):
        """Convert to objects form using `Trial.build`."""
        trials = Trial.build(exp_config[1])
        assert list(map(lambda x: x.to_dict(), trials)) == exp_config[1]

    def test_value_equal(self, exp_config):
        """Compare Param objects using __eq__"""
        trials = Trial.build(exp_config[1])

        assert trials[0]._params[0] == Trial.Param(**exp_config[1][0]["params"][0])
        assert trials[0]._params[1] != Trial.Param(**exp_config[1][0]["params"][0])

    def test_str_trial(self, trial_config):
        """Test representation of `Trial`."""
        t = Trial(**trial_config)
        assert (
            str(t) == "Trial(experiment='supernaedo2-dendi', status='completed', "
            "params=/decoding_layer:lstm_with_attention,/encoding_layer:gru)"
        )

    def test_str_value(self, trial_config):
        """Test representation of `Trial.Value`."""
        t = Trial(**trial_config)
        assert (
            str(t._params[1])
            == "Param(name='/encoding_layer', type='categorical', value='gru')"
        )

    def test_invalid_result(self, trial_config):
        """Test that invalid objectives cannot be set"""
        t = Trial(**trial_config)

        # Make sure valid ones pass
        t.results = [
            Trial.Result(name="asfda", type="constraint", value=None),
            Trial.Result(name="asfdb", type="objective", value=1e-5),
        ]

        # No results at all
        with pytest.raises(ValueError) as exc:
            t.results = []
        assert "No objective found" in str(exc.value)

        # No objectives
        with pytest.raises(ValueError) as exc:
            t.results = [Trial.Result(name="asfda", type="constraint", value=None)]
        assert "No objective found" in str(exc.value)

        # Bad objective type
        with pytest.raises(ValueError) as exc:
            t.results = [
                Trial.Result(name="asfda", type="constraint", value=None),
                Trial.Result(name="asfdb", type="objective", value=None),
            ]
        assert "Results must contain" in str(exc.value)

    def test_objective_property(self):
        """Check property `Trial.objective`."""
        base_trial = {"results": []}
        base_trial["results"].append({"name": "a", "type": "gradient", "value": 0.5})

        # 0 objective in `results` list
        trial = Trial(**base_trial)

        assert trial.objective is None

        # 1 results in `results` list
        expected = Trial.Result(name="b", type="objective", value=42)

        base_trial["results"].append({"name": "b", "type": "objective", "value": 42})
        trial = Trial(**base_trial)

        assert trial.objective == expected

        # >1 results in `results` list
        base_trial["results"].append({"name": "c", "type": "objective", "value": 42})
        trial = Trial(**base_trial)

        assert trial.objective == expected

    def test_gradient_property(self):
        """Check property `Trial.gradient`."""
        base_trial = {"results": []}

        # 0 result exist
        trial = Trial(**base_trial)
        assert trial.gradient is None

        # 0 result of type 'gradient' exist
        base_trial["results"].append({"name": "a", "type": "objective", "value": 10})
        trial = Trial(**base_trial)

        assert trial.gradient is None

        # 1 result of type 'gradient' exist
        expected = Trial.Result(name="b", type="gradient", value=5)

        base_trial["results"].append({"name": "b", "type": "gradient", "value": 5})
        trial = Trial(**base_trial)

        assert trial.gradient == expected

        # >1 gradient result
        base_trial["results"].append(
            {"name": "c", "type": "gradient", "value": [12, 15]}
        )
        trial = Trial(**base_trial)

        assert trial.gradient == expected

    def test_constraints_property(self):
        """Tests the property for accessing constraints"""
        base_trial = {"results": []}

        # 0 result exist
        trial = Trial(**base_trial)
        assert trial.constraints == []

        # 0 result of type 'constraints' exist
        base_trial["results"].append({"name": "a", "type": "objective", "value": 10})
        trial = Trial(**base_trial)

        assert trial.constraints == []

        # 1 result of type 'constraint' exist
        expected = [Trial.Result(name="b", type="constraint", value=5)]

        base_trial["results"].append({"name": "b", "type": "constraint", "value": 5})
        trial = Trial(**base_trial)

        assert expected == trial.constraints

        # > 1 results of type 'constraint' exist
        expected = [
            Trial.Result(name="b", type="constraint", value=5),
            Trial.Result(name="c", type="constraint", value=20),
        ]

        base_trial["results"].append({"name": "c", "type": "constraint", "value": 20})
        trial = Trial(**base_trial)

        assert expected == trial.constraints

    def test_statistics_property(self):
        """Tests the property for accessing statistics"""
        base_trial = {"results": []}

        # 0 result exist
        trial = Trial(**base_trial)
        assert trial.statistics == []

        # 0 result of type 'statistic' exist
        base_trial["results"].append({"name": "a", "type": "objective", "value": 10})
        trial = Trial(**base_trial)

        assert trial.statistics == []

        # 1 result of type 'statistic' exist
        expected = [Trial.Result(name="b", type="statistic", value=5)]

        base_trial["results"].append({"name": "b", "type": "statistic", "value": 5})
        trial = Trial(**base_trial)

        assert expected == trial.statistics

        # > 1 results of type 'statistic' exist
        expected = [
            Trial.Result(name="b", type="statistic", value=5),
            Trial.Result(name="c", type="statistic", value=20),
        ]

        base_trial["results"].append({"name": "c", "type": "statistic", "value": 20})
        trial = Trial(**base_trial)

        assert expected == trial.statistics

    def test_params_repr_property(self, trial_config):
        """Check property `Trial.params_repr`."""
        t = Trial(**trial_config)
        assert (
            Trial.format_params(t._params)
            == "/decoding_layer:lstm_with_attention,/encoding_layer:gru"
        )
        assert (
            Trial.format_params(t._params, sep="\n")
            == "/decoding_layer:lstm_with_attention\n/encoding_layer:gru"
        )

        t = Trial()
        assert Trial.format_params(t._params) == ""

    def test_hash_name_property(self, trial_config):
        """Check property `Trial.hash_name`."""
        t = Trial(**trial_config)
        assert t.hash_name == "eebf174e46dae012521c12985b652cff"

        t = Trial()
        with pytest.raises(ValueError) as exc:
            t.hash_name
        assert "params" in str(exc.value)

    def test_param_name_property(self, trial_config):
        """Check property `Trial.hash_params`."""
        trial_config["params"].append(
            {"name": "/max_epoch", "type": "fidelity", "value": "1"}
        )
        t1 = Trial(**trial_config)
        trial_config["params"][-1]["value"] = "2"  # changing the fidelity
        t2 = Trial(**trial_config)
        assert t1.hash_name != t2.hash_name
        assert t1.hash_params == t2.hash_params

    def test_hash_ignore_experiment_is_deprecated(self, trial_config):
        """Check property `Trial.compute_trial_hash(ignore_experiment=True)` is deprecated."""
        trial_config["params"].append(
            {"name": "/max_epoch", "type": "fidelity", "value": "1"}
        )
        t1 = Trial(**trial_config)
        trial_config["experiment"] = "test"  # changing the experiment name
        t2 = Trial(**trial_config)
        with warnings.catch_warnings(record=True) as w:
            assert t1.hash_name == t2.hash_name
            assert t1.hash_params == t2.hash_params
            assert len(w) == 0

        with pytest.deprecated_call():
            assert Trial.compute_trial_hash(
                t1, ignore_experiment=False
            ) != Trial.compute_trial_hash(t2, ignore_experiment=False)

        with pytest.deprecated_call():
            assert Trial.compute_trial_hash(
                t1, ignore_experiment=True
            ) == Trial.compute_trial_hash(t2, ignore_experiment=True)

    def test_hash_ignore_lie(self, trial_config):
        """Check property `Trial.compute_trial_hash(ignore_lie=True)`."""
        trial_config["params"].append(
            {"name": "/max_epoch", "type": "fidelity", "value": "1"}
        )
        t1 = Trial(**trial_config)
        # Add a lie
        trial_config["results"].append({"name": "lie", "type": "lie", "value": 1})
        t2 = Trial(**trial_config)
        assert t1.hash_name != t2.hash_name
        assert t1.hash_params == t2.hash_params
        assert Trial.compute_trial_hash(
            t1, ignore_lie=True
        ) == Trial.compute_trial_hash(t2, ignore_lie=True)

    def test_hash_ignore_parent(self, trial_config):
        """Check property `Trial.compute_trial_hash(ignore_parent=True)`."""
        trial_config["params"].append(
            {"name": "/max_epoch", "type": "fidelity", "value": "1"}
        )
        t1 = Trial(**trial_config)
        trial_config["parent"] = 0
        t2 = Trial(**trial_config)
        assert t1.hash_name != t2.hash_name
        assert t1.hash_params == t2.hash_params
        assert Trial.compute_trial_hash(
            t1, ignore_parent=True
        ) == Trial.compute_trial_hash(t2, ignore_parent=True)

    def test_full_name_property(self, trial_config):
        """Check property `Trial.full_name`."""
        t = Trial(**trial_config)
        assert t.full_name == ".decoding_layer:lstm_with_attention-.encoding_layer:gru"

        t = Trial()
        with pytest.raises(ValueError) as exc:
            t.full_name
        assert "params" in str(exc.value)

    def test_higher_shape_id_is_same(self):
        """Check if a Trial with a shape > 1 has the same id once it has been through the DB."""
        x = {"name": "/x", "value": [1, 2], "type": "real"}
        trial = Trial(params=[x])
        assert (
            trial.id == Trial(**bson.BSON.decode(bson.BSON.encode(trial.to_dict()))).id
        )

    def test_equal(self, trial_config):
        """Check that two trials are equal based on id"""

        trial_config["params"].append(
            {"name": "/max_epoch", "type": "fidelity", "value": "1"}
        )
        t1 = Trial(**trial_config)

        def change_attr(attrname, attrvalue):
            t2 = Trial(**trial_config)
            assert t1 == t2
            setattr(t2, attrname, attrvalue)
            return t2

        t2 = change_attr("parent", 0)
        assert t1 != t2

        params = copy.deepcopy(t1._params)
        params[-1].value = "2"
        t2 = change_attr("_params", params)
        assert t1 != t2

        t2 = change_attr("exp_working_dir", "whatever")
        assert t1 == t2

        t2 = change_attr("status", "broken")
        assert t1 == t2

    def test_no_exp_working_dir(self):
        trial = Trial()

        with pytest.raises(RuntimeError, match="Cannot infer trial's working_dir"):
            trial.working_dir

    def test_working_dir(self, tmp_path, params):
        trial = Trial(experiment=0, exp_working_dir=tmp_path, params=params, parent=1)
        assert trial.working_dir == os.path.join(tmp_path, trial.id)
        assert trial.get_working_dir() == os.path.join(tmp_path, trial.id)

        trial._params.append(Trial.Param(name="/epoch", type="fidelity", value=1))

        assert trial.id != trial.hash_params
        assert trial.get_working_dir(
            ignore_fidelity=True, ignore_lie=True, ignore_parent=True
        ) == os.path.join(tmp_path, trial.hash_params)

        assert trial.get_working_dir(ignore_parent=True) != trial.working_dir

    def test_branch_empty(self, base_trial):
        """Test that branching with no args is only copying"""
        branched_trial = base_trial.branch()
        assert branched_trial.experiment is base_trial.experiment
        assert branched_trial is not base_trial
        assert branched_trial.status == "new"
        assert branched_trial.start_time is None
        assert branched_trial.end_time is None
        assert branched_trial.heartbeat is None
        assert branched_trial.params == base_trial.params
        assert branched_trial.objective is None
        assert branched_trial.parent == base_trial.id
        assert branched_trial.exp_working_dir == base_trial.exp_working_dir
        assert branched_trial.id != base_trial.id

    def test_branch_base_attr(self, base_trial):
        """Test branching with base attributes (not params)"""
        branched_trial = base_trial.branch(status="interrupted")
        assert branched_trial.status != base_trial.status
        assert branched_trial.status == "interrupted"
        assert branched_trial.params == base_trial.params
        assert branched_trial.parent == base_trial.id
        assert branched_trial.exp_working_dir == base_trial.exp_working_dir
        assert branched_trial.id != base_trial.id

    def test_branch_params(self, base_trial):
        """Test branching with params"""
        branched_trial = base_trial.branch(status="interrupted", params={"/y": [3, 0]})
        assert branched_trial.status != base_trial.status
        assert branched_trial.status == "interrupted"
        assert branched_trial.params != base_trial.params
        assert branched_trial.params == {"/x": [1, 2], "/y": [3, 0]}
        assert branched_trial.parent == base_trial.id
        assert branched_trial.exp_working_dir == base_trial.exp_working_dir
        assert branched_trial.id != base_trial.id

    def test_branch_new_params(self, base_trial):
        """Test branching with params that are not in base trial"""
        with pytest.raises(
            ValueError, match="Some parameters are not part of base trial: {'/z': 0}"
        ):
            base_trial.branch(params={"/z": 0})

    def test_execution_interval_property(self, trial_config):
        """Check property `Trial.execution_interval`."""
        t = Trial(**trial_config)
        assert t.execution_interval == (150, "2017-11-23T00:00:00")

        t = Trial()
        assert t.execution_interval is None

    def test_execution_interval_property_no_end_time(self, trial_config_no_end_time):
        """Check property `Trial.execution_interval` with no end time"""
        t = Trial(**trial_config_no_end_time)
        assert t.end_time is None
        assert t.heartbeat is not None
        assert t.execution_interval == (t.start_time, t.heartbeat)

    def test_duration_property(self, trial_config_with_correct_start_time):
        """Check property `Trial.duration`."""
        t = Trial(**trial_config_with_correct_start_time)
        assert t.duration == datetime.timedelta(seconds=3600)

        t = Trial()
        assert t.duration == datetime.timedelta()

    def test_duration_property_no_end_time(self, trial_config_no_end_time):
        """Check property `Trial.duration` with no end time"""
        t = Trial(**trial_config_no_end_time)
        assert t.end_time is None
        assert t.heartbeat is not None
        assert t.duration == t.heartbeat - t.start_time
