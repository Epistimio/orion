#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the setup command."""
import builtins
import os

import yaml

import orion.core.cli
import orion.core.io.resolve_config as resolve_config


def _mock_input(*args):
    return "allo"


def _mock_empty_input(*args):
    return ""


def test_creation_when_not_existing(monkeypatch, tmp_path):
    """Test if a configuration file is created when it does not exist."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(resolve_config, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input)

    try:
        os.remove(config_path)
    except FileNotFoundError:
        pass

    orion.core.cli.main(["setup"])

    assert os.path.exists(config_path)

    with open(config_path, 'r') as output:
        content = yaml.safe_load(output)

    assert content == {"database": {"type": "allo", "name": "allo", "host": "allo"}}


def test_creation_when_exists(monkeypatch, tmp_path):
    """Test if the configuration file is overwritten when it exists."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(resolve_config, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input)

    dump = {"database": {"type": "allo2", "name": "allo2", "host": "allo2"}}

    with open(config_path, 'w') as output:
        yaml.dump(dump, output)

    orion.core.cli.main(["setup"])

    with open(config_path, 'r') as output:
        content = yaml.safe_load(output)

    assert content != dump


def test_defaults(monkeypatch, tmp_path):
    """Test if the default values are used when nothing user enters nothing."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(resolve_config, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_empty_input)

    orion.core.cli.main(["setup"])

    with open(config_path, 'r') as output:
        content = yaml.safe_load(output)

    assert content == {"database": {"type": "mongodb", "name": "test", "host": "localhost"}}
