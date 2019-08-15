#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.storage`."""

import pytest

from orion.storage.base import BaseStorageProtocol, FailedUpdate, get_storage, Storage
from orion.core.utils.tests import OrionState


def test_create_experiment():
    """test update heartbeat"""
    pass


def test_fetch_experiments():
    """test update heartbeat"""
    pass


def test_register_trial():
    """test update heartbeat"""
    pass


def test_register_lie():
    """test update heartbeat"""
    pass


def test_reserve_trial():
    """test update heartbeat"""
    pass


def test_fetch_trials():
    """test update heartbeat"""
    pass


def test_fetch_experiment_trials():
    """test update heartbeat"""
    pass


def test_get_trial():
    """test update heartbeat"""
    pass


def test_fetch_lost_trials():
    """test update heartbeat"""
    pass


def test_retrieve_result():
    """test update heartbeat"""
    pass


def test_push_trial_results():
    """test update heartbeat"""
    pass


def test_change_status_success(exp_config_file):
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


def test_change_status_failed_update(exp_config_file):
    """Successfully find new trials in db and reserve one at 'random'."""

    def check_status_change(new_status):
        with OrionState(from_yaml=exp_config_file) as cfg:
            trial = cfg.get_trial(0)
            assert trial is not None, 'Was not able to retrieve trial for test'

            with pytest.raises(FailedUpdate) as exc:
                trial.status = new_status
                get_storage().set_trial_status(trial, status=new_status)

    check_status_change('completed')
    check_status_change('broken')
    check_status_change('reserved')
    check_status_change('interrupted')
    check_status_change('suspended')
    check_status_change('new')


def test_fetch_pending_trials():
    """test update heartbeat"""
    pass


def test_fetch_noncompleted_trials():
    """test update heartbeat"""
    pass


def test_fetch_completed_trials():
    """test update heartbeat"""
    pass


def test_count_completed_trials():
    """test update heartbeat"""
    pass


def test_count_broken_trials():
    """test update heartbeat"""
    pass


def test_update_heartbeat():
    """test update heartbeat"""
    pass

