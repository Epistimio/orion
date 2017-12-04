#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`metaopt.worker.experiment`."""

import datetime
import getpass

import pytest

from metaopt.worker.experiment import Experiment


@pytest.fixture()
def with_user_tsirif(monkeypatch):
    """Make ``getpass.getuser()`` return ``'tsirif'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'tsirif')


@pytest.fixture()
def with_user_bouthilx(monkeypatch):
    """Make ``getpass.getuser()`` return ``'bouthilx'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'bouthilx')


@pytest.fixture()
def random_dt(monkeypatch):
    """Make ``datetime.datetime.utcnow()`` return an arbitrary date."""
    random_dt = datetime.datetime(1903, 4, 25, 0, 0, 0)

    class MockDatetime(datetime.datetime):
        @classmethod
        def utcnow(cls):
            return random_dt

    monkeypatch.setattr(datetime, 'datetime', MockDatetime)
    return random_dt


class TestInitExperiment(object):
    """Create new Experiment instance."""

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_new_experiment_due_to_name(self, create_db_instance, random_dt):
        """Hit user name, but exp_name does not hit the db, create new entry."""
        exp = Experiment('supernaekei')
        assert exp._init_done is False
        assert exp._db is create_db_instance
        assert exp._id is None
        assert exp.name == 'supernaekei'
        assert exp.refers is None
        assert exp.metadata['user'] == 'tsirif'
        assert exp.metadata['datetime'] == random_dt
        assert len(exp.metadata) == 2
        assert exp.pool_size is None
        assert exp.max_trials is None
        assert exp.status is None
        assert exp.algorithms is None
        with pytest.raises(AttributeError):
            exp.this_is_not_in_config = 5

    @pytest.mark.usefixtures("with_user_bouthilx")
    def test_new_experiment_due_to_username(self, create_db_instance, random_dt):
        """Hit exp_name, but user's name does not hit the db, create new entry."""
        exp = Experiment('supernaedo2')
        assert exp._init_done is False
        assert exp._db is create_db_instance
        assert exp._id is None
        assert exp.name == 'supernaedo2'
        assert exp.refers is None
        assert exp.metadata['user'] == 'bouthilx'
        assert exp.metadata['datetime'] == random_dt
        assert len(exp.metadata) == 2
        assert exp.pool_size is None
        assert exp.max_trials is None
        assert exp.status is None
        assert exp.algorithms is None
        with pytest.raises(AttributeError):
            exp.this_is_not_in_config = 5

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_existing_experiment(self, create_db_instance, exp_config):
        """Hit exp_name + user's name in the db, fetch most recent entry."""
        exp = Experiment('supernaedo2')
        assert exp._init_done is False
        assert exp._db is create_db_instance
        assert exp._id == exp_config[0][1]['_id']
        assert exp.name == exp_config[0][1]['name']
        assert exp.refers == exp_config[0][1]['refers']
        assert exp.metadata == exp_config[0][1]['metadata']
        assert exp.pool_size == exp_config[0][1]['pool_size']
        assert exp.max_trials == exp_config[0][1]['max_trials']
        assert exp.status == exp_config[0][1]['status']
        assert exp.algorithms == exp_config[0][1]['algorithms']
        with pytest.raises(AttributeError):
            exp.this_is_not_in_config = 5


@pytest.mark.usefixtures("create_db_instance", "with_user_tsirif")
class TestConfigProperty(object):
    """Get and set experiment's configuration, finilize initialization process."""

    def test_get_before_init_has_hit(self, exp_config, random_dt):
        """Return a configuration dict according to an experiment object.

        Assuming that experiment's (exp's name, user's name) has hit the database.
        """
        exp = Experiment('supernaedo2')
        exp_config[0][1].pop('_id')
        assert exp.config == exp_config[0][1]

    def test_get_before_init_no_hit(self, exp_config, random_dt):
        """Return a configuration dict according to an experiment object.

        Before initialization is done, it can be the case that the pair (`name`,
        user's name) has not hit the database. return a yaml compliant form
        of current state, to be used with :mod:`metaopt.resolve_config`.
        """
        exp = Experiment('supernaekei')
        cfg = exp.config
        assert cfg['name'] == 'supernaekei'
        assert cfg['refers'] is None
        assert cfg['metadata']['user'] == 'tsirif'
        assert cfg['metadata']['datetime'] == random_dt
        assert len(cfg['metadata']) == 2
        assert cfg['pool_size'] is None
        assert cfg['max_trials'] is None
        assert cfg['status'] is None
        assert cfg['algorithms'] is None
        assert len(cfg) == 7

    def test_set_before_init_with_diffs(self, exp_config):
        pass

    def test_set_before_init_no_diffs(self, exp_config):
        pass

    def test_get_after_init(self, exp_config):
        """Return a configuration dict according to an experiment object.

        Before initialization is done, it can be the case that the pair (`name`,
        user's name) has not hit the database. return a yaml compliant form
        of current state, to be used with :mod:`metaopt.resolve_config`.
        """
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp.config = exp_config[0][1]
        assert exp.config == exp_config[0][1]

    def test_set_after_init(self, exp_config):
        """Cannot set a configuration after init (currently)."""
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp.config = exp_config[0][1]
        with pytest.raises(ValueError) as exc_info:
            exp.config = exp_config[0][1]
        assert 'Cannot set' in str(exc_info.value)
