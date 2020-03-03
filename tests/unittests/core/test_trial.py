#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.trial`."""
import bson
import numpy
import pytest

from orion.core.worker.trial import Trial


class TestTrial(object):
    """Test Trial object and class."""

    def test_init_empty(self):
        """Initialize empty trial."""
        t = Trial()
        assert t.experiment is None
        assert t.status == 'new'
        assert t.worker is None
        assert t.submit_time is None
        assert t.start_time is None
        assert t.end_time is None
        assert t.results == []
        assert t.params == {}
        assert t.working_dir is None

    def test_init_full(self, exp_config):
        """Initialize with a dictionary with complete specification."""
        t = Trial(**exp_config[1][1])
        assert t.experiment == exp_config[1][1]['experiment']
        assert t.status == exp_config[1][1]['status']
        assert t.worker == exp_config[1][1]['worker']
        assert t.submit_time == exp_config[1][1]['submit_time']
        assert t.start_time == exp_config[1][1]['start_time']
        assert t.end_time == exp_config[1][1]['end_time']
        assert list(map(lambda x: x.to_dict(), t.results)) == exp_config[1][1]['results']
        assert t.results[0].name == exp_config[1][1]['results'][0]['name']
        assert t.results[0].type == exp_config[1][1]['results'][0]['type']
        assert t.results[0].value == exp_config[1][1]['results'][0]['value']
        assert list(map(lambda x: x.to_dict(), t._params)) == exp_config[1][1]['params']
        assert t.working_dir is None

    def test_higher_shapes_not_ndarray(self):
        """Test that `numpy.ndarray` values are converted to list."""
        value = numpy.zeros([3])
        expected = value.tolist()
        params = [dict(name='/x', type='real', value=value)]
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
            Trial(ispii='iela')

    def test_value_bad_init(self):
        """Other than `Trial.Value.__slots__` are not allowed in __init__ too."""
        with pytest.raises(AttributeError):
            Trial.Value(ispii='iela')

    def test_param_bad_init(self):
        """Other than `Trial.Param.__slots__` are not allowed in __init__ too."""
        with pytest.raises(AttributeError):
            Trial.Param(ispii='iela')

    def test_result_bad_init(self):
        """Other than `Trial.Result.__slots__` are not allowed in __init__ too."""
        with pytest.raises(AttributeError):
            Trial.Result(ispii='iela')

    def test_not_allowed_status(self):
        """Other than `Trial.allowed_stati` are not allowed in `Trial.status`."""
        t = Trial()
        with pytest.raises(ValueError):
            t.status = 'asdf'
        with pytest.raises(ValueError):
            t = Trial(status='ispi')

    def test_value_not_allowed_type(self):
        """Other than `Trial.Result.allowed_types` are not allowed in `Trial.Result.type`.

        Same for `Trial.Param`.
        """
        with pytest.raises(ValueError):
            v = Trial.Result(name='asfda', type='hoho')
        v = Trial.Result()
        with pytest.raises(ValueError):
            v.type = 'asfda'

        with pytest.raises(ValueError):
            v = Trial.Param(name='asfda', type='hoho')
        v = Trial.Param()
        with pytest.raises(ValueError):
            v.type = 'asfda'

    @pytest.mark.usefixtures("clean_db")
    def test_convertion_to_dict(self, exp_config):
        """Convert to dictionary form for database using ``dict``."""
        t = Trial(**exp_config[1][1])
        assert t.to_dict() == exp_config[1][1]

    @pytest.mark.usefixtures("clean_db")
    def test_build_trials(self, exp_config):
        """Convert to objects form using `Trial.build`."""
        trials = Trial.build(exp_config[1])
        assert list(map(lambda x: x.to_dict(), trials)) == exp_config[1]

    @pytest.mark.usefixtures("clean_db")
    def test_value_equal(self, exp_config):
        """Compare Param objects using __eq__"""
        trials = Trial.build(exp_config[1])

        assert trials[0]._params[0] == Trial.Param(**exp_config[1][0]['params'][0])
        assert trials[0]._params[1] != Trial.Param(**exp_config[1][0]['params'][0])

    def test_str_trial(self, exp_config):
        """Test representation of `Trial`."""
        t = Trial(**exp_config[1][1])
        assert str(t) == "Trial(experiment='supernaedo2-dendi', status='completed', "\
                         "params=/decoding_layer:lstm_with_attention,/encoding_layer:gru)"

    def test_str_value(self, exp_config):
        """Test representation of `Trial.Value`."""
        t = Trial(**exp_config[1][1])
        assert (str(t._params[1]) ==
                "Param(name='/encoding_layer', type='categorical', value='gru')")

    def test_invalid_result(self, exp_config):
        """Test that invalid objectives cannot be set"""
        t = Trial(**exp_config[1][1])

        # Make sure valid ones pass
        t.results = [Trial.Result(name='asfda', type='constraint', value=None),
                     Trial.Result(name='asfdb', type='objective', value=1e-5)]

        # No results at all
        with pytest.raises(ValueError) as exc:
            t.results = []
        assert 'No objective found' in str(exc.value)

        # No objectives
        with pytest.raises(ValueError) as exc:
            t.results = [Trial.Result(name='asfda', type='constraint', value=None)]
        assert 'No objective found' in str(exc.value)

        # Bad objective type
        with pytest.raises(ValueError) as exc:
            t.results = [Trial.Result(name='asfda', type='constraint', value=None),
                         Trial.Result(name='asfdb', type='objective', value=None)]
        assert 'Results must contain' in str(exc.value)

    def test_objective_property(self, exp_config):
        """Check property `Trial.objective`."""
        # 1 results in `results` list
        t = Trial(**exp_config[1][1])
        assert isinstance(t.objective, Trial.Result)
        assert t.objective.name == 'yolo'
        assert t.objective.type == 'objective'
        assert t.objective.value == 10

        # 0 results in `results` list
        tmp = exp_config[1][1]['results'].pop(0)
        t = Trial(**exp_config[1][1])
        assert t.objective is None
        exp_config[1][1]['results'].append(tmp)

        # >1 results in `results` list
        exp_config[1][1]['results'].append(dict(name='yolo2',
                                                type='objective',
                                                value=12))
        t = Trial(**exp_config[1][1])
        assert isinstance(t.objective, Trial.Result)
        assert t.objective.name == 'yolo'
        assert t.objective.type == 'objective'
        assert t.objective.value == 10
        tmp = exp_config[1][1]['results'].pop()

    def test_gradient_property(self, exp_config):
        """Check property `Trial.gradient`."""
        # 1 results in `results` list
        t = Trial(**exp_config[1][1])
        assert isinstance(t.gradient, Trial.Result)
        assert t.gradient.name == 'naedw_grad'
        assert t.gradient.type == 'gradient'
        assert t.gradient.value == [5, 3]

        # 0 results in `results` list
        tmp = exp_config[1][1]['results'].pop()
        t = Trial(**exp_config[1][1])
        assert t.gradient is None
        exp_config[1][1]['results'].append(tmp)

        # >1 results in `results` list
        exp_config[1][1]['results'].append(dict(name='yolo2',
                                                type='gradient',
                                                value=[12, 15]))
        t = Trial(**exp_config[1][1])
        assert isinstance(t.gradient, Trial.Result)
        assert t.gradient.name == 'naedw_grad'
        assert t.gradient.type == 'gradient'
        assert t.gradient.value == [5, 3]
        tmp = exp_config[1][1]['results'].pop()

    def test_params_repr_property(self, exp_config):
        """Check property `Trial.params_repr`."""
        t = Trial(**exp_config[1][1])
        assert Trial.format_params(t._params) == \
            "/decoding_layer:lstm_with_attention,/encoding_layer:gru"
        assert Trial.format_params(t._params, sep='\n') == \
            "/decoding_layer:lstm_with_attention\n/encoding_layer:gru"

        t = Trial()
        assert Trial.format_params(t._params) == ""

    def test_hash_name_property(self, exp_config):
        """Check property `Trial.hash_name`."""
        t = Trial(**exp_config[1][1])
        assert t.hash_name == "ebcf6c6c8604f96444af1c3e519aea7f"

        t = Trial()
        with pytest.raises(ValueError) as exc:
            t.hash_name
        assert 'params' in str(exc.value)

    def test_param_name_property(self, exp_config):
        """Check property `Trial.hash_params`."""
        exp_config[1][1]['params'].append({'name': '/max_epoch', 'type': 'fidelity', 'value': '1'})
        t1 = Trial(**exp_config[1][1])
        exp_config[1][1]['params'][-1]['value'] = '2'  # changing the fidelity
        t2 = Trial(**exp_config[1][1])
        assert t1.hash_name != t2.hash_name
        assert t1.hash_params == t2.hash_params

    def test_hash_ignore_experiment(self, exp_config):
        """Check property `Trial.compute_trial_hash(ignore_experiment=True)`."""
        exp_config[1][1]['params'].append({'name': '/max_epoch', 'type': 'fidelity', 'value': '1'})
        t1 = Trial(**exp_config[1][1])
        exp_config[1][1]['experiment'] = 'test'  # changing the experiment name
        t2 = Trial(**exp_config[1][1])
        assert t1.hash_name != t2.hash_name
        assert t1.hash_params != t2.hash_params
        assert (Trial.compute_trial_hash(t1, ignore_experiment=True) ==
                Trial.compute_trial_hash(t2, ignore_experiment=True))

    def test_hash_ignore_lie(self, exp_config):
        """Check property `Trial.compute_trial_hash(ignore_lie=True)`."""
        exp_config[1][1]['params'].append({'name': '/max_epoch', 'type': 'fidelity', 'value': '1'})
        t1 = Trial(**exp_config[1][1])
        # Add a lie
        exp_config[1][1]['results'].append({'name': 'lie', 'type': 'lie', 'value': 1})
        t2 = Trial(**exp_config[1][1])
        assert t1.hash_name != t2.hash_name
        assert t1.hash_params == t2.hash_params
        assert (Trial.compute_trial_hash(t1, ignore_lie=True) ==
                Trial.compute_trial_hash(t2, ignore_lie=True))

    def test_full_name_property(self, exp_config):
        """Check property `Trial.full_name`."""
        t = Trial(**exp_config[1][1])
        assert t.full_name == ".decoding_layer:lstm_with_attention-.encoding_layer:gru"

        t = Trial()
        with pytest.raises(ValueError) as exc:
            t.full_name
        assert 'params' in str(exc.value)

    def test_higher_shape_id_is_same(self):
        """Check if a Trial with a shape > 1 has the same id once it has been through the DB."""
        x = {'name': '/x', 'value': [1, 2], 'type': 'real'}
        trial = Trial(params=[x])
        assert trial.id == Trial(**bson.BSON.decode(bson.BSON.encode(trial.to_dict()))).id
