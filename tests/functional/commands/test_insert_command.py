#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the default commands for demo purposes."""
import os
import subprocess

import pytest


@pytest.mark.usefixtures("clean_db")
def test_insert_invalid_experiment(database, monkeypatch):
    """Test the insertion of an invalid experiment"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    try:
        process = subprocess.Popen(["orion", "insert", "-n", "dumb_experiment",
                                    "-c", "./orion_config_random.yaml", "./black_box.py", "-x=1"])
        process.wait()
    except ValueError as err:
        assert str(err) == "No experiment with given name inside database, can't insert"
        pass


@pytest.mark.usefixtures("only_experiments_db")
def test_insert_single_trial(database, monkeypatch):
    """Try to insert a single trial"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen(["orion", "insert", "-n", "test_insert_normal",
                                "-c", "./orion_config_random.yaml", "./black_box.py", "-x=1"])

    rcode = process.wait()
    assert rcode == 0

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
def test_insert_single_trial_default_value(database, monkeypatch):
    """Try to insert a single trial using a default value"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen(["orion", "insert", "-n", "test_insert_normal",
                                "-c", "./orion_config_random.yaml", "./black_box.py"])

    rcode = process.wait()
    assert rcode == 0

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
def test_insert_with_no_default_value(database, monkeypatch):
    """Try to insert a single trial by omitting a namespace with no default value"""
    try:
        monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
        process = subprocess.Popen(["orion", "insert", "-n", "test_insert_normal",
                                    "-c", "./orion_config_random.yaml", "./black_box.py"])

        process.wait()
    except ValueError as err:
        assert str(err) == "Dimension /x is unspecified and has no default value"
        pass


@pytest.mark.usefixtures("only_experiments_db")
def test_insert_with_incorrect_namespace(database, monkeypatch):
    """Try to insert a single trial with a namespace not inside the experiment space"""
    try:
        monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
        process = subprocess.Popen(["orion", "insert", "-n", "test_insert_normal",
                                    "-c", "./orion_config_random.yaml", "./black_box.py", "-p=4"])

        process.wait()
    except ValueError as err:
        assert str(err) == "Found namespace outside of experiment space : /p"
        pass


@pytest.mark.usefixtures("only_experiments_db")
def test_insert_with_outside_bound_value(database, monkeypatch):
    """Try to insert a single trial with value outside the distribution's interval"""
    try:
        monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
        process = subprocess.Popen(["orion", "insert", "-n", "test_insert_two_hyperparameters",
                                    "-c", "./orion_config_random.yaml",
                                    "./black_box.py", "-x=4", "-y=100"])

        process.wait()
    except ValueError as err:
        assert str(err) == "Value 100 is outside of dimension's prior interval"
        pass


@pytest.mark.usefixtures("only_experiments_db")
def test_insert_two_hyperparameters(database, monkeypatch):
    """Try to insert a single trial with two hyperparameters"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen(["orion", "insert", "-n", "test_insert_two_hyperparameters",
                                "-c", "./orion_config_random.yaml",
                                "./black_box.py", "-x=1", "-y=2"])

    rcode = process.wait()
    assert rcode == 0

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
