#!/usr/bin/env python
"""Perform a functional test of the default commands for demo purposes."""
import os

import pytest

import orion.core.cli


def get_user_corneau():
    """Return user corneau (to mock getpass.getuser)"""
    return "corneau"


def test_insert_invalid_experiment(storage, monkeypatch, capsys):
    """Test the insertion of an invalid experiment"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    returncode = orion.core.cli.main(
        [
            "insert",
            "-n",
            "dumb_experiment",
            "-c",
            "./orion_config_random.yaml",
            "./black_box.py",
            "-x=1",
        ]
    )

    assert returncode == 1

    captured = capsys.readouterr().err

    assert "Error: No experiment with given name 'dumb_experiment'"


@pytest.mark.usefixtures("only_experiments_db", "version_XYZ")
def test_insert_single_trial(storage, monkeypatch, script_path):
    """Try to insert a single trial"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    orion.core.cli.main(
        [
            "insert",
            "-n",
            "test_insert_normal",
            "-c",
            "./orion_config_random.yaml",
            "blabla",
            "-x=1",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "test_insert_normal"}))

    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp

    trials = list(storage.fetch_trials(uid=exp["_id"]))

    assert len(trials) == 1

    trial = trials[0]

    assert trial.status == "new"
    assert trial.params["/x"] == 1


@pytest.mark.usefixtures("only_experiments_db", "version_XYZ")
def test_insert_single_trial_default_value(storage, monkeypatch):
    """Try to insert a single trial using a default value"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    orion.core.cli.main(
        [
            "insert",
            "-n",
            "test_insert_normal",
            "-c",
            "./orion_config_random.yaml",
            "./black_box.py",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "test_insert_normal"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp

    trials = list(storage.fetch_trials(uid=exp["_id"]))

    assert len(trials) == 1

    trial = trials[0]

    assert trial.status == "new"
    assert trial.params["/x"] == 1


@pytest.mark.usefixtures("only_experiments_db")
def test_insert_with_no_default_value(monkeypatch):
    """Try to insert a single trial by omitting a namespace with no default value"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    with pytest.raises(ValueError) as exc_info:
        orion.core.cli.main(
            [
                "insert",
                "-n",
                "test_insert_missing_default_value",
                "-c",
                "./orion_config_random.yaml",
                "./black_box.py",
            ]
        )

    assert "Dimension /x is unspecified and has no default value" in str(exc_info.value)


@pytest.mark.usefixtures("only_experiments_db")
def test_insert_with_incorrect_namespace(monkeypatch):
    """Try to insert a single trial with a namespace not inside the experiment space"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    with pytest.raises(ValueError) as exc_info:
        orion.core.cli.main(
            [
                "insert",
                "-n",
                "test_insert_normal",
                "-c",
                "./orion_config_random.yaml",
                "./black_box.py",
                "-p=4",
            ]
        )

    assert "Found namespace outside of experiment space : /p" in str(exc_info.value)


@pytest.mark.usefixtures("only_experiments_db")
def test_insert_with_outside_bound_value(monkeypatch):
    """Try to insert a single trial with value outside the distribution's interval"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    with pytest.raises(ValueError) as exc_info:
        orion.core.cli.main(
            [
                "insert",
                "-n",
                "test_insert_two_hyperparameters",
                "-c",
                "./orion_config_random.yaml",
                "./black_box.py",
                "-x=4",
                "-y=100",
            ]
        )
    assert "Value 100 is outside of" in str(exc_info.value)


@pytest.mark.usefixtures("only_experiments_db", "version_XYZ")
def test_insert_two_hyperparameters(storage, monkeypatch):
    """Try to insert a single trial with two hyperparameters"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)
    orion.core.cli.main(
        [
            "insert",
            "-n",
            "test_insert_two_hyperparameters",
            "-c",
            "./orion_config_random.yaml",
            "./black_box.py",
            "-x=1",
            "-y=2",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "test_insert_two_hyperparameters"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp

    trials = list(storage.fetch_trials(uid=exp["_id"]))

    assert len(trials) == 1

    trial = trials[0]

    assert trial.status == "new"
    assert trial.params["/x"] == 1
    assert trial.params["/y"] == 2


def test_insert_with_version(storage, monkeypatch, script_path):
    """Try to insert a single trial inside a specific version"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "-n",
            "experiment",
            "-c",
            "./orion_config_random.yaml",
            script_path,
            "-x~normal(0,1)",
        ]
    )
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--enable-evc",
            "-n",
            "experiment",
            "-c",
            "./orion_config_random.yaml",
            script_path,
            "-x~normal(0,1)",
            "-y~+normal(0,1)",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "experiment", "version": 1}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp

    trials = list(storage.fetch_trials(uid=exp["_id"]))
    assert len(trials) == 0

    orion.core.cli.main(
        [
            "insert",
            "-n",
            "experiment",
            "--version",
            "1",
            "-c",
            "./orion_config_random.yaml",
            script_path,
            "-x=1",
        ]
    )

    trials = list(storage.fetch_trials(uid=exp["_id"]))

    assert len(trials) == 1


def test_no_args(capsys):
    """Test that help is printed when no args are given."""
    with pytest.raises(SystemExit):
        orion.core.cli.main(["insert"])

    captured = capsys.readouterr().out

    assert "usage:" in captured
    assert "Traceback" not in captured


def test_no_name(capsys):
    """Try to run the command without providing an experiment name"""
    returncode = orion.core.cli.main(["insert", "--version", "1"])
    assert returncode == 1

    captured = capsys.readouterr().err

    assert captured == "Error: No name provided for the experiment.\n"
