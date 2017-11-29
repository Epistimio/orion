#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`metaopt.worker.experiment`."""

from datetime import datetime
import getpass

import pytest

from metaopt.worker.experiment import Experiment


@pytest.fixture()
def with_user_tsirif(monkeypatch):
    """Make ``getpass.getuser()`` return ``'tsirif'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'tsirif')


@pytest.fixture()
def with_user_xavier(monkeypatch):
    """Make ``getpass.getuser()`` return ``'bouthilx'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'bouthilx')


@pytest.fixture()
def random_dt(monkeypatch):
    """Make ``datetime.datetime.utcnow()`` return an arbitrary date."""
    random_dt = datetime(1903, 4, 25, 0, 0, 0)
    monkeypatch.setattr(datetime, 'utcnow', lambda: random_dt)
    return random_dt


class TestInitExperiment(object):
    """Create new Experiment instance."""

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_new_experiment_due_to_name(self, random_dt):
        """exp_name does not hit the db, create new entry."""
        exp = Experiment('supernaekei')
        assert exp.name == 'supernaekei'
        assert exp.user == 'tsirif'
        assert exp.datetime == random_dt
        assert exp.metadata == dict()
        assert exp.references == list()
        assert exp.pool_size is None
        assert exp.max_trials is None
        assert exp.status == 'new'
        assert exp.stats == dict()
        assert exp.algorithms == dict()

    @pytest.mark.usefixtures("with_user_xavier")
    def test_new_experiment_due_to_username(self, random_dt):
        """User's name does not hit the db, create new entry."""
        exp = Experiment('supernaedo2')
        assert exp.name == 'supernaedo2'
        assert exp.user == 'xavier'
        assert exp.datetime == random_dt
        assert exp.metadata == dict()
        assert exp.references == list()
        assert exp.pool_size is None
        assert exp.max_trials is None
        assert exp.status == 'new'
        assert exp.stats == dict()
        assert exp.algorithms == dict()

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_existing_experiment(self, exp_config):
        """exp_name + user's name hit the db, fetch most recent entry."""
        exp = Experiment('supernaedo2')
        assert exp.name == exp_config[0][1]['exp_name']
        assert exp.user == exp_config[0][1]['user']
        assert exp.datetime == exp_config[0][1]['datetime']
        assert exp.metadata == exp_config[0][1]['metadata']
        assert exp.references == exp_config[0][1]['references']
        assert exp.pool_size == exp_config[0][1]['pool_size']
        assert exp.max_trials == exp_config[0][1]['max_trials']
        assert exp.status == exp_config[0][1]['status']
        assert exp.stats == exp_config[0][1]['stats']
        assert exp.algorithms == exp_config[0][1]['algorithms']
