#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for demo purposes."""
import os
import subprocess

import numpy
import pytest

from orion.core.io.database import Database
from orion.core.worker import workon
from orion.core.worker.experiment import Experiment


@pytest.mark.usefixtures("clean_db")
def test_demo_with_default_algo_cli_config_only(database, monkeypatch):
    """Check that random algorithm is used, when no algo is chosen explicitly."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')
    process = subprocess.Popen(["orion", "hunt", "-n", "default_algo",
                                "--max-trials", "30",
                                "./black_box.py", "-x~uniform(-50, 50)"])
    rcode = process.wait()
    assert rcode == 0

    exp = list(database.experiments.find({'name': 'default_algo'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    assert exp['name'] == 'default_algo'
    assert exp['pool_size'] == 10
    assert exp['max_trials'] == 30
    assert exp['status'] == 'done'
    assert exp['algorithms'] == {'random': {}}
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'orion_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~uniform(-50, 50)']


@pytest.mark.usefixtures("clean_db")
def test_demo(database, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen(["orion", "hunt", "--config", "./orion_config.yaml",
                                "./black_box.py", "-x~uniform(-50, 50)"])
    rcode = process.wait()
    assert rcode == 0

    exp = list(database.experiments.find({'name': 'voila_voici'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    assert exp['name'] == 'voila_voici'
    assert exp['pool_size'] == 1
    assert exp['max_trials'] == 100
    assert exp['status'] == 'done'
    assert exp['algorithms'] == {'gradient_descent': {'learning_rate': 0.1}}
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'orion_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~uniform(-50, 50)']

    trials = list(database.trials.find({'experiment': exp_id}))
    assert len(trials) < 15
    assert trials[-1]['status'] == 'completed'
    for result in trials[-1]['results']:
        assert result['type'] != 'constraint'
        if result['type'] == 'objective':
            assert abs(result['value'] - 23.4) < 1e-6
            assert result['name'] == 'example_objective'
        elif result['type'] == 'gradient':
            res = numpy.asarray(result['value'])
            assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-7
            assert result['name'] == 'example_gradient'
    params = trials[-1]['params']
    assert len(params) == 1
    assert params[0]['name'] == '/x'
    assert params[0]['type'] == 'real'
    assert (params[0]['value'] - 34.56789) < 1e-5


@pytest.mark.usefixtures("clean_db")
def test_demo_two_workers(database, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    processes = []
    for _ in range(2):
        process = subprocess.Popen(["orion", "hunt", "-n", "two_workers_demo",
                                    "--config", "./orion_config_random.yaml",
                                    "./black_box.py", "-x~norm(34, 3)"])
        processes.append(process)

    for process in processes:
        rcode = process.wait()
        assert rcode == 0

    exp = list(database.experiments.find({'name': 'two_workers_demo'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    assert exp['name'] == 'two_workers_demo'
    assert exp['pool_size'] == 2
    assert exp['max_trials'] == 400
    assert exp['status'] == 'done'
    assert exp['algorithms'] == {'random': {}}
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'orion_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~norm(34, 3)']

    trials = list(database.trials.find({'experiment': exp_id}))
    for trial in trials:
        assert trial['status'] == 'completed'
    assert len(trials) >= 400
    assert len(trials) <= 402
    params = trials[-1]['params']
    assert len(params) == 1
    assert params[0]['name'] == '/x'
    assert params[0]['type'] == 'real'


@pytest.mark.usefixtures("clean_db")
def test_workon(database):
    """Test scenario having a configured experiment already setup."""
    try:
        Database(of_type='MongoDB', name='orion_test',
                 username='user', password='pass')
    except (TypeError, ValueError):
        pass
    experiment = Experiment('voila_voici')
    config = experiment.configuration
    config['algorithms'] = {
        'gradient_descent': {
            'learning_rate': 0.1
            }
        }
    config['pool_size'] = 1
    config['max_trials'] = 100
    config['metadata']['user_script'] = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "black_box.py"))
    config['metadata']['user_args'] = ["-x~uniform(-50, 50)"]
    experiment.configure(config)

    workon(experiment)

    exp = list(database.experiments.find({'name': 'voila_voici'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    assert exp['name'] == 'voila_voici'
    assert exp['pool_size'] == 1
    assert exp['max_trials'] == 100
    assert exp['status'] == 'done'
    assert exp['algorithms'] == {'gradient_descent': {'learning_rate': 0.1}}
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~uniform(-50, 50)']

    trials = list(database.trials.find({'experiment': exp_id}))
    assert len(trials) < 15
    assert trials[-1]['status'] == 'completed'
    for result in trials[-1]['results']:
        assert result['type'] != 'constraint'
        if result['type'] == 'objective':
            assert abs(result['value'] - 23.4) < 1e-6
            assert result['name'] == 'example_objective'
        elif result['type'] == 'gradient':
            res = numpy.asarray(result['value'])
            assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-7
            assert result['name'] == 'example_gradient'
    params = trials[-1]['params']
    assert len(params) == 1
    assert params[0]['name'] == '/x'
    assert params[0]['type'] == 'real'
    assert (params[0]['value'] - 34.56789) < 1e-5
