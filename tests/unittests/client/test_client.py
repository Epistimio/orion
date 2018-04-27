#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.client`."""

from importlib import reload
import json

import pytest

from orion import client


@pytest.fixture()
def data():
    """Return serializable data."""
    return "this is datum"


class TestReportResults(object):
    """Check functionality and edge cases of `report_results` helper interface."""

    def test_with_no_env(self, monkeypatch, capsys, data):
        """Test without having set the appropriate environmental variable.

        Then: It should print `data` parameter instead to stdout.
        """
        monkeypatch.delenv('ORION_RESULTS_PATH', raising=False)
        reloaded_client = reload(client)

        assert reloaded_client.IS_ORION_ON is False
        assert reloaded_client.RESULTS_FILENAME is None
        assert reloaded_client._HAS_REPORTED_RESULTS is False

        reloaded_client.report_results(data)
        out, err = capsys.readouterr()
        assert reloaded_client._HAS_REPORTED_RESULTS is True
        assert out == data + '\n'
        assert err == ''

    def test_with_correct_env(self, monkeypatch, capsys, tmpdir, data):
        """Check that a file with correct data will be written to an existing
        file in a legit path.
        """
        path = str(tmpdir.join('naedw.txt'))
        with open(path, mode='w'):
            pass
        monkeypatch.setenv('ORION_RESULTS_PATH', path)
        reloaded_client = reload(client)

        assert reloaded_client.IS_ORION_ON is True
        assert reloaded_client.RESULTS_FILENAME == path
        assert reloaded_client._HAS_REPORTED_RESULTS is False

        reloaded_client.report_results(data)
        out, err = capsys.readouterr()
        assert reloaded_client._HAS_REPORTED_RESULTS is True
        assert out == ''
        assert err == ''

        with open(path, mode='r') as results_file:
            res = json.load(results_file)
        assert res == data

    def test_with_env_set_but_no_file_exists(self, monkeypatch, tmpdir, data):
        """Check that a Warning will be raised at import time,
        if environmental is set but does not correspond to an existing file.
        """
        path = str(tmpdir.join('naedw.txt'))
        monkeypatch.setenv('ORION_RESULTS_PATH', path)

        with pytest.raises(RuntimeWarning) as exc:
            reload(client)

        assert "existing file" in str(exc.value)

    def test_call_interface_twice(self, monkeypatch, data):
        """Check that a Warning will be raised at call time,
        if function has already been called once.
        """
        monkeypatch.delenv('ORION_RESULTS_PATH', raising=False)
        reloaded_client = reload(client)

        reloaded_client.report_results(data)
        with pytest.raises(RuntimeWarning) as exc:
            reloaded_client.report_results(data)

        assert "already reported" in str(exc.value)
        assert reloaded_client.IS_ORION_ON is False
        assert reloaded_client.RESULTS_FILENAME is None
        assert reloaded_client._HAS_REPORTED_RESULTS is True
