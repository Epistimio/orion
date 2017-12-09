#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`metaopt.worker.trial`."""

import pytest

from metaopt.worker.trial import Trial


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
        assert t.params == []

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
        assert list(map(lambda x: x.to_dict(), t.params)) == exp_config[1][1]['params']

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

    def test_str_trial(self, exp_config):
        """Test representation of `Trial`."""
        t = Trial(**exp_config[1][2])
        assert str(t) == "Trial(experiment='supernaedo2', "\
                         "status='completed', params.value=['gru', 'lstm_with_attention'])"

    def test_str_value(self, exp_config):
        """Test representation of `Trial.Value`."""
        t = Trial(**exp_config[1][2])
        assert str(t.params[0]) == "Param(name='encoding_layer', "\
                                   "type='enum', value='gru')"
