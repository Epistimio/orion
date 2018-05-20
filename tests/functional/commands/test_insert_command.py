#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the default commands for demo purposes."""
import os

import pytest
import yaml

import orion.core.cli


@pytest.fixture(scope='module')
def exp_config():
    """Load an example database."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))
    return exp_config


@pytest.fixture()
def clean_db(database):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()


def get_user_corneau():
    """Return user corneau (to mock getpass.getuser)"""
    return "corneau"


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_insert_invalid_experiment(database, monkeypatch):
    """Test the insertion of an invalid experiment"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    with pytest.raises(ValueError) as exc_info:
        orion.core.cli._main(["insert", "-n", "dumb_experiment",
                              "-c", "./orion_config_random.yaml", "./black_box.py", "-x=1"])

    assert ("No experiment with given name 'dumb_experiment' for user 'corneau'"
            in str(exc_info.value))


@pytest.mark.usefixtures("only_experiments_db")
@pytest.mark.usefixtures("null_db_instances")
def test_insert_single_trial(database, monkeypatch):
    """Try to insert a single trial"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    orion.core.cli._main(["insert", "-n", "test_insert_normal",
                          "-c", "./orion_config_random.yaml", "./black_box.py", "-x=1"])

    exp = list(database.experiments.find({"name": "test_insert_normal"}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp

    trials = list(database.trials.find({"experiment": exp['_id']}))

    assert len(trials) == 1

    trial = trials[0]

    assert trial['status'] == 'new'
    assert trial['params'][0]['value'] == 1


@pytest.mark.usefixtures("only_experiments_db")
@pytest.mark.usefixtures("null_db_instances")
def test_insert_single_trial_default_value(database, monkeypatch):
    """Try to insert a single trial using a default value"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    orion.core.cli._main(["insert", "-n", "test_insert_normal",
                          "-c", "./orion_config_random.yaml", "./black_box.py"])

    exp = list(database.experiments.find({"name": "test_insert_normal"}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp

    trials = list(database.trials.find({"experiment": exp['_id']}))

    assert len(trials) == 1

    trial = trials[0]

    assert trial['status'] == 'new'
    assert trial['params'][0]['value'] == 1


@pytest.mark.usefixtures("only_experiments_db")
@pytest.mark.usefixtures("null_db_instances")
def test_insert_with_no_default_value(database, monkeypatch):
    """Try to insert a single trial by omitting a namespace with no default value"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    with pytest.raises(ValueError) as exc_info:
        orion.core.cli._main(["insert", "-n", "test_insert_missing_default_value",
                              "-c", "./orion_config_random.yaml", "./black_box.py"])

    assert "Dimension /x is unspecified and has no default value" in str(exc_info.value)


@pytest.mark.usefixtures("only_experiments_db")
@pytest.mark.usefixtures("null_db_instances")
def test_insert_with_incorrect_namespace(database, monkeypatch):
    """Try to insert a single trial with a namespace not inside the experiment space"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    with pytest.raises(ValueError) as exc_info:
        orion.core.cli._main(["insert", "-n", "test_insert_normal",
                              "-c", "./orion_config_random.yaml",
                              "./black_box.py", "-p=4"])

    assert "Found namespace outside of experiment space : /p" in str(exc_info.value)


@pytest.mark.usefixtures("only_experiments_db")
@pytest.mark.usefixtures("null_db_instances")
def test_insert_with_outside_bound_value(database, monkeypatch):
    """Try to insert a single trial with value outside the distribution's interval"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)

    with pytest.raises(ValueError) as exc_info:
        orion.core.cli._main(["insert", "-n", "test_insert_two_hyperparameters",
                              "-c", "./orion_config_random.yaml",
                              "./black_box.py", "-x=4", "-y=100"])
    assert "Value 100 is outside of" in str(exc_info.value)


@pytest.mark.usefixtures("only_experiments_db")
@pytest.mark.usefixtures("null_db_instances")
def test_insert_two_hyperparameters(database, monkeypatch):
    """Try to insert a single trial with two hyperparameters"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setattr("getpass.getuser", get_user_corneau)
    orion.core.cli._main(["insert", "-n", "test_insert_two_hyperparameters",
                          "-c", "./orion_config_random.yaml", "./black_box.py", "-x=1", "-y=2"])

    exp = list(database.experiments.find({"name": "test_insert_two_hyperparameters"}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp

    trials = list(database.trials.find({"experiment": exp['_id']}))

    assert len(trials) == 1

    trial = trials[0]

    assert trial['status'] == 'new'
    assert trial['params'][0]['value'] == 1
    assert trial['params'][1]['value'] == 2
