#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.cli.checks`."""
import pytest

import orion.core
from orion.core.cli.checks.creation import CreationStage
from orion.core.cli.checks.operations import OperationsStage
from orion.core.cli.checks.presence import PresenceStage
from orion.core.io.database.mongodb import MongoDB
import orion.core.io.experiment_builder as experiment_builder
import orion.core.utils.backward as backward
from orion.core.utils.exceptions import CheckError


@pytest.fixture
def config():
    """Return a basic database configuration."""
    return {'database': {'host': 'localhost', 'type': 'mongodb', 'name': 'user'}}


@pytest.fixture
def presence():
    """Return a PresenceStage instance."""
    return PresenceStage([])


@pytest.fixture
def creation(create_db_instance):
    """Return a CreationStage instance."""
    stage = CreationStage(None)
    stage.instance = create_db_instance
    return stage


@pytest.fixture
def operation(creation):
    """Return an OperationStage instance."""
    return OperationsStage(creation)


@pytest.fixture
def clean_test(database):
    """Remove the test collection from database."""
    database.test.drop()


def test_check_default_config_pass(monkeypatch, presence, config):
    """Check if the default config test works."""
    def mock_default_config(self):
        return config

    monkeypatch.setattr(orion.core.config.__class__, 'to_dict', mock_default_config)

    result, msg = presence.check_default_config()

    assert result == "Success"
    assert msg == ""
    assert presence.db_config == config['storage']['database']


def test_check_default_config_skip(monkeypatch, presence):
    """Check if test returns skip if no default config is found."""
    def mock_default_config(self):
        return {}

    monkeypatch.setattr(orion.core.config.__class__, 'to_dict', mock_default_config)

    result, msg = presence.check_default_config()
    assert result == "Skipping"
    assert "No" in msg
    assert presence.db_config == {}


def test_config_file_config_pass(monkeypatch, presence, config):
    """Check if test passes with valid configuration."""
    def mock_file_config(cmdargs):
        backward.update_db_config(config)
        return config

    monkeypatch.setattr(experiment_builder, "get_cmd_config", mock_file_config)

    result, msg = presence.check_configuration_file()

    assert result == "Success"
    assert msg == ""
    assert presence.db_config == config['storage']['database']


def test_config_file_fails_missing_config(monkeypatch, presence, config):
    """Check if test fails with missing configuration."""
    def mock_file_config(cmdargs):
        return {}

    monkeypatch.setattr(experiment_builder, "get_cmd_config", mock_file_config)

    status, msg = presence.check_configuration_file()

    assert status == "Skipping"
    assert "Missing" in msg
    assert presence.db_config == {}


def test_config_file_fails_missing_database(monkeypatch, presence, config):
    """Check if test fails with missing database configuration."""
    def mock_file_config(cmdargs):
        return {'algorithm': 'asha'}

    monkeypatch.setattr(experiment_builder, "get_cmd_config", mock_file_config)

    status, msg = presence.check_configuration_file()

    assert status == "Skipping"
    assert "No database" in msg
    assert presence.db_config == {}


def test_config_file_fails_missing_value(monkeypatch, presence, config):
    """Check if test fails with missing value in database configuration."""
    def mock_file_config(cmdargs):
        return {'storage': {'database': {}}}

    monkeypatch.setattr(experiment_builder, "get_cmd_config", mock_file_config)

    status, msg = presence.check_configuration_file()

    assert status == "Skipping"
    assert "No configuration" in msg
    assert presence.db_config == {}


def test_config_file_skips(monkeypatch, presence, config):
    """Check if test skips when another configuration is present."""
    def mock_file_config(self):
        return {}

    presence.db_config = config['database']
    monkeypatch.setattr(experiment_builder, "get_cmd_config", mock_file_config)

    result, msg = presence.check_configuration_file()

    assert result == "Skipping"
    assert presence.db_config == config['database']


@pytest.mark.usefixtures('null_db_instances')
def test_creation_pass(presence, config):
    """Check if test passes with valid database configuration."""
    presence.db_config = config['database']
    creation = CreationStage(presence)

    result, msg = creation.check_database_creation()

    assert result == "Success"
    assert msg == ""
    assert creation.instance is not None


@pytest.mark.usefixtures('null_db_instances')
def test_creation_fails(monkeypatch, presence, config):
    """Check if test fails when not connected."""
    presence.db_config = config['database']
    creation = CreationStage(presence)

    monkeypatch.setattr(MongoDB, "is_connected", False)

    with pytest.raises(CheckError) as ex:
        creation.check_database_creation()

    assert "failed" in str(ex.value)


def test_operation_write_pass(operation):
    """Check if test passes when write operation works."""
    result, msg = operation.check_write()

    assert result == "Success"
    assert msg == ""


def test_operation_write_fails(monkeypatch, operation):
    """Check if test fails when write operation fails."""
    def mock_write(one, two):
        raise RuntimeError("Not working")

    monkeypatch.setattr(operation.c_stage.instance, "write", mock_write)

    with pytest.raises(CheckError) as ex:
        operation.check_write()

    assert "Not working" in str(ex.value)


def test_operation_read_pass(operation, clean_test):
    """Check if test passes when read operation works."""
    operation.c_stage.instance.write('test', {'index': 'value'})
    result, msg = operation.check_read()

    assert result == "Success"
    assert msg == ""


def test_operation_read_fail_not_working(monkeypatch, operation):
    """Check if test fails when read operation fails."""
    def mock_read(one, two):
        raise RuntimeError("Not working")

    monkeypatch.setattr(operation.c_stage.instance, "read", mock_read)

    with pytest.raises(CheckError) as ex:
        operation.check_read()

    assert "Not working" in str(ex.value)


def test_operation_read_fail_unexpected_value(operation, clean_test):
    """Check if test fails on unexpected read value."""
    operation.c_stage.instance.write('test', {'index': 'value2'})

    with pytest.raises(CheckError) as ex:
        operation.check_read()

    assert "value" in str(ex.value)


def test_operation_count_pass(operation, clean_test):
    """Check if test passes when count operation works."""
    operation.c_stage.instance.write('test', {'index': 'value'})
    result, msg = operation.check_count()

    assert result == "Success"
    assert msg == ""


def test_operation_count_fails(monkeypatch, operation, clean_test):
    """Check if test fails when count operation fails."""
    operation.c_stage.instance.write('test', {'index': 'value'})
    operation.c_stage.instance.write('test', {'index': 'value'})
    with pytest.raises(CheckError) as ex:
        operation.check_count()

    assert "2" in str(ex.value)


def test_operation_remove_pass(operation, clean_test):
    """Check if test passes when remove operation works."""
    operation.c_stage.instance.write('test', {'index': 'value'})
    result, msg = operation.check_remove()

    assert result == "Success"
    assert msg == ""


def test_operation_remove_fails(monkeypatch, operation, clean_test, database):
    """Check if test fails when remove operation fails."""
    operation.c_stage.instance.write('test', {'index': 'value'})
    operation.c_stage.instance.write('test', {'index': 'value'})

    def mock_remove(one, two):
        database.test.delete_one({'index': 'value'})

    monkeypatch.setattr(operation.c_stage.instance, "remove", mock_remove)

    with pytest.raises(CheckError) as ex:
        operation.check_remove()

    assert "1" in str(ex.value)
