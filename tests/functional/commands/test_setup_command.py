#!/usr/bin/env python
"""Perform a functional test of the setup command."""
import builtins
import os

import yaml

import orion.core
import orion.core.cli
from orion.core.io.database import database_factory


class _mock_input:
    """Mock `input` to return a series of strings."""

    def __init__(self, values):
        """Set the series of values"""
        self.values = values

    def __call__(self, *args):
        """Pop one value at each"""
        return self.values.pop(0)


def test_creation_when_not_existing(monkeypatch, tmp_path):
    """Test if a configuration file is created when it does not exist."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(["mongodb", "host", "name"]))

    try:
        os.remove(config_path)
    except FileNotFoundError:
        pass

    orion.core.cli.main(["db", "setup"])

    assert os.path.exists(config_path)

    with open(config_path) as output:
        content = yaml.safe_load(output)

    assert content == {"database": {"type": "mongodb", "name": "name", "host": "host"}}


def test_creation_when_exists(monkeypatch, tmp_path):
    """Test if the configuration file is overwritten when it exists."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(
        builtins, "input", _mock_input(["y", "mongodb", "host", "name"])
    )

    dump = {"database": {"type": "allo2", "name": "allo2", "host": "allo2"}}

    with open(config_path, "w") as output:
        yaml.dump(dump, output)

    orion.core.cli.main(["db", "setup"])

    with open(config_path) as output:
        content = yaml.safe_load(output)

    assert content != dump


def test_stop_creation_when_exists(monkeypatch, tmp_path):
    """Test if the configuration file is overwritten when it exists."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(["n"]))

    dump = {"database": {"type": "allo2", "name": "allo2", "host": "allo2"}}

    with open(config_path, "w") as output:
        yaml.dump(dump, output)

    orion.core.cli.main(["db", "setup"])

    with open(config_path) as output:
        content = yaml.safe_load(output)

    assert content == dump


def test_invalid_database(monkeypatch, tmp_path, capsys):
    """Test if command prompt loops when invalid database is typed."""
    invalid_db_names = [
        "invalid database",
        "invalid database again",
        "2383ejdd",
        "another invalid database",
    ]
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(
        builtins,
        "input",
        _mock_input(
            [
                *invalid_db_names,
                "mongodb",
                "the host",
                "the name",
            ]
        ),
    )

    orion.core.cli.main(["db", "setup"])

    with open(config_path) as output:
        content = yaml.safe_load(output)

    assert content == {
        "database": {"type": "mongodb", "name": "the name", "host": "the host"}
    }

    captured_output = capsys.readouterr().out
    for invalid_db_name in invalid_db_names:
        assert (
            "Unexpected value: {}. Must be one of: {}\n".format(
                invalid_db_name,
                ", ".join(sorted(database_factory.get_classes().keys())),
            )
            in captured_output
        )


def test_defaults(monkeypatch, tmp_path):
    """Test if the default values are used when nothing user enters nothing."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(["", "", ""]))

    orion.core.cli.main(["db", "setup"])

    with open(config_path) as output:
        content = yaml.safe_load(output)

    assert content == {
        "database": {"type": "mongodb", "name": "orion", "host": "localhost"}
    }


def test_ephemeraldb(monkeypatch, tmp_path):
    """Test if config content is written for an ephemeraldb."""
    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(["ephemeraldb"]))

    orion.core.cli.main(["db", "setup"])

    with open(config_path) as output:
        content = yaml.safe_load(output)

    assert content == {"database": {"type": "ephemeraldb"}}


def test_pickleddb(monkeypatch, tmp_path):
    """Test if config content is written for an pickleddb."""
    host = "my_pickles.db"

    config_path = str(tmp_path) + "/tmp_config.yaml"
    monkeypatch.setattr(orion.core, "DEF_CONFIG_FILES_PATHS", [config_path])
    monkeypatch.setattr(builtins, "input", _mock_input(["pickleddb", host]))

    orion.core.cli.main(["db", "setup"])

    with open(config_path) as output:
        content = yaml.safe_load(output)

    assert content == {"database": {"type": "pickleddb", "host": host}}
