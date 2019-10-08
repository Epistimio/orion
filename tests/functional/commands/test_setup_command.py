#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the setup command."""
import builtins
import os

import yaml

import orion.core
import orion.core.cli


class _mock_input:
    """Mock `input` to return a serie of strings."""

    def __init__(self, values):
        """Set the serie of values"""
        self.values = values

    def __call__(self, *args):
        """Pop one value at each"""
        return self.values.pop(0)


def test_creation_when_not_existing(monkeypatch, tmp_path):
    """Test if a configuration file is created when it does not exist."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(['type', 'name', 'host']))

    try:
        os.remove(config_path)
    except FileNotFoundError:
        pass

    orion.core.cli.main(["db", "setup"])

    assert os.path.exists(config_path)

    with open(config_path, 'r') as output:
        content = yaml.safe_load(output)

    assert content == {"database": {"type": "type", "name": "name", "host": "host"}}


def test_creation_when_exists(monkeypatch, tmp_path):
    """Test if the configuration file is overwritten when it exists."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(['y', 'type', 'name', 'host']))

    dump = {"database": {"type": "allo2", "name": "allo2", "host": "allo2"}}

    with open(config_path, 'w') as output:
        yaml.dump(dump, output)

    orion.core.cli.main(["db", "setup"])

    with open(config_path, 'r') as output:
        content = yaml.safe_load(output)

    assert content != dump


def test_stop_creation_when_exists(monkeypatch, tmp_path):
    """Test if the configuration file is overwritten when it exists."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(['n']))

    dump = {"database": {"type": "allo2", "name": "allo2", "host": "allo2"}}

    with open(config_path, 'w') as output:
        yaml.dump(dump, output)

    orion.core.cli.main(["db", "setup"])

    with open(config_path, 'r') as output:
        content = yaml.safe_load(output)

    assert content == dump


def test_defaults(monkeypatch, tmp_path):
    """Test if the default values are used when nothing user enters nothing."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(['', '', '']))

    orion.core.cli.main(["db", "setup"])

    with open(config_path, 'r') as output:
        content = yaml.safe_load(output)

    assert content == {"database": {"type": "mongodb", "name": "test", "host": "localhost"}}
