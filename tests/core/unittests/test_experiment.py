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


@pytest.fixture()
def new_config(random_dt):
    """Create a configuration that will not hit the database."""
    new_config = dict(
        name='supernaekei',
        # refers is missing on purpose
        metadata={'user': 'tsirif',
                  'datetime': random_dt,
                  'mopt_version': 0.1,
                  'user_script': 'abs_path/to_yoyoy.py',
                  'user_config': 'abs_path/hereitis.yaml',
                  'user_args': '--mini-batch~uniform(32, 256)',
                  'user_vcs': 'git',
                  'user_version': 0.8,
                  'user_commit_hash': 'fsa7df7a8sdf7a8s7'},
        pool_size=10,
        max_trials=1000,
        algorithms={'bayesian_scikit': {'yoyo': 5, 'alpha': 0.8}},
        # the following should be ignored by the setting function:
        # status is not settable:
        status='asdfafa',
        # attrs starting with '_' also
        _id='fasdfasfa',
        # and in general anything which is not in Experiment's slots
        something_to_be_ignored='asdfa'
        )
    return new_config


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
        assert exp.configuration == exp_config[0][1]

    def test_get_before_init_no_hit(self, exp_config, random_dt):
        """Return a configuration dict according to an experiment object.

        Before initialization is done, it can be the case that the pair (`name`,
        user's name) has not hit the database. return a yaml compliant form
        of current state, to be used with :mod:`metaopt.resolve_config`.
        """
        exp = Experiment('supernaekei')
        cfg = exp.configuration
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

    def test_good_set_before_init_hit_with_diffs(self, exp_config):
        """Trying to set, and differences were found from the config pulled from db.

        In this case:
        1. Force renaming of experiment, prompt user for new name.
        2. Fork from experiment with previous name. New experiments refers to the
           old one, if user wants to.
        3. Overwrite elements with the ones from input.

        .. warning:: Currently, not implemented.
        """
        new_config = {'pool_size': 8}
        exp = Experiment('supernaedo2')
        with pytest.raises(NotImplementedError):
            exp.configure(new_config)

    def test_good_set_before_init_hit_no_diffs_exc_max_trials(self, exp_config):
        """Trying to set, and NO differences were found from the config pulled from db.

        Everything is normal, nothing changes. Experiment is resumed,
        perhaps with more trials to evaluate (the only exception is 'max_trials').
        """
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp_config[0][1]['max_trials'] = 5000
        exp_config[0][1]['status'] = 'new'
        exp.configure(exp_config[0][1])
        assert exp.configuration == exp_config[0][1]
        exp_config[0][1]['max_trials'] = 1000  # don't ruin resource
        exp_config[0][1]['status'] = 'running'

    def test_good_set_before_init_no_hit(self, random_dt, database, new_config):
        """Trying to set, overwrite everything from input."""
        exp = Experiment(new_config['name'])
        exp.configure(new_config)
        assert exp._init_done is True
        found_config = list(database.experiments.find({'name': 'supernaekei',
                                                       'metadata.user': 'tsirif'}))
        assert len(found_config) == 1
        _id = found_config[0].pop('_id')
        assert _id != 'fasdfasfa'
        assert exp._id == _id
        new_config['refers'] = None
        new_config['status'] = 'new'
        new_config.pop('_id')
        new_config.pop('something_to_be_ignored')
        assert found_config[0] == new_config
        assert exp.name == new_config['name']
        assert exp.refers is None
        assert exp.metadata == new_config['metadata']
        assert exp.pool_size == new_config['pool_size']
        assert exp.max_trials == new_config['max_trials']
        assert exp.status == new_config['status']
        #  assert exp.algorithms == new_config['algorithms']

    def test_inconsistent_1_set_before_init_no_hit(self, random_dt, new_config):
        """Test inconsistent configuration because of name."""
        exp = Experiment(new_config['name'])
        new_config['name'] = 'asdfaa'
        with pytest.raises(ValueError) as exc_info:
            exp.configure(new_config)
        assert 'inconsistent' in str(exc_info.value)

    def test_inconsistent_2_set_before_init_no_hit(self, random_dt, new_config):
        """Test inconsistent configuration because of user."""
        exp = Experiment(new_config['name'])
        new_config['metadata']['user'] = 'asdfaa'
        with pytest.raises(ValueError) as exc_info:
            exp.configure(new_config)
        assert 'inconsistent' in str(exc_info.value)

    def test_inconsistent_3_set_before_init_no_hit(self, random_dt, new_config):
        """Test inconsistent configuration because of datetime."""
        exp = Experiment(new_config['name'])
        new_config['metadata']['datetime'] = 123
        with pytest.raises(ValueError) as exc_info:
            exp.configure(new_config)
        assert 'inconsistent' in str(exc_info.value)

    def test_get_after_init_plus_hit_no_diffs(self, exp_config):
        """Return a configuration dict according to an experiment object.

        Before initialization is done, it can be the case that the pair (`name`,
        user's name) has not hit the database. return a yaml compliant form
        of current state, to be used with :mod:`metaopt.resolve_config`.
        """
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp.configure(exp_config[0][1])
        assert exp._init_done is True
        exp_config[0][1]['status'] = 'new'
        assert exp.configuration == exp_config[0][1]
        exp_config[0][1]['status'] = 'running'  # don't ruin resource

    def test_try_set_after_init(self, exp_config):
        """Cannot set a configuration after init (currently)."""
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp.configure(exp_config[0][1])
        assert exp._init_done is True
        with pytest.raises(RuntimeError) as exc_info:
            exp.configure(exp_config[0][1])
        assert 'cannot reset' in str(exc_info.value)

    def test_after_init_algorithms_are_objects(self, exp_config):
        """Attribute exp.algorithms become objects after init."""
        pass

    def test_after_init_refers_are_objects(self, exp_config):
        """Attribute exp.refers become objects after init."""
        pass
