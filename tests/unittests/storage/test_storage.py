#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.storage`."""

import pytest

from orion.core.utils.tests import OrionState
from orion.storage.base import FailedUpdate, get_storage


@pytest.mark.parametrize('storage', ['legacy'])
class StorageTest:
    def test_create_experiment(self, storage):
        """Test create experiment"""
        pass

    def test_fetch_experiments(self, storage):
        """Test fetch expriments"""
        pass

    def test_register_trial(self, storage):
        """Test register trial"""
        pass

    def test_register_lie(self, storage):
        """Test register lie"""
        pass

    def test_reserve_trial(self, storage):
        """Test reserve trial"""
        pass

    def test_fetch_trials(self, storage):
        """Test fetch trials"""
        pass

    def test_fetch_experiment_trials(self, storage):
        """Test fetch experiment trials"""
        pass

    def test_get_trial(self, storage):
        """Test get trial"""
        pass

    def test_fetch_lost_trials(self, storage):
        """Test update heartbeat"""
        pass

    def test_retrieve_result(self, storage):
        """Test retrieve result"""
        pass

    def test_push_trial_results(self, storage):
        """Test push trial results"""
        pass

    def test_change_status_success(self, storage, exp_config_file):
        """Change the status of a Trial"""
        def check_status_change(new_status):
            with OrionState(from_yaml=exp_config_file) as cfg:
                trial = cfg.get_trial(0)
                assert trial is not None, 'was not able to retrieve trial for test'

                get_storage().set_trial_status(trial, status=new_status)
                assert trial.status == new_status, \
                    'Trial status should have been updated locally'

                trial = get_storage().get_trial(trial)
                assert trial.status == new_status, \
                    'Trial status should have been updated in the storage'

        check_status_change('completed')
        check_status_change('broken')
        check_status_change('reserved')
        check_status_change('interrupted')
        check_status_change('suspended')
        check_status_change('new')

    def test_change_status_failed_update(self, storage, exp_config_file):
        """Successfully find new trials in db and reserve one at 'random'."""
        def check_status_change(new_status):
            with OrionState(from_yaml=exp_config_file) as cfg:
                trial = cfg.get_trial(0)
                assert trial is not None, 'Was not able to retrieve trial for test'

                with pytest.raises(FailedUpdate):
                    trial.status = new_status
                    get_storage().set_trial_status(trial, status=new_status)

        check_status_change('completed')
        check_status_change('broken')
        check_status_change('reserved')
        check_status_change('interrupted')
        check_status_change('suspended')

    def test_fetch_pending_trials(self, storage):
        """Test fetch pending trials"""
        pass

    def test_fetch_noncompleted_trials(self, storage):
        """Test fetch non completed trials"""
        pass

    def test_fetch_completed_trials(self, storage):
        """Test fetch completed trials"""
        pass

    def test_count_completed_trials(self, storage):
        """Test count completed trials"""
        pass

    def test_count_broken_trials(self, storage):
        """Test count broken trials"""
        pass

    def test_update_heartbeat(self, storage):
        """Test update heartbeat"""
        pass
