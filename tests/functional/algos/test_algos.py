#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for algos included with orion."""
import os

import pytest
import yaml

import orion.core.cli
from orion.storage.base import get_storage


config_files = ['random_config.yaml']
fidelity_config_files = ['random_config.yaml', 'asha_config.yaml']
fidelity_only_config_files = list(set(fidelity_config_files) - set(config_files))


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
@pytest.mark.parametrize('config_file', fidelity_only_config_files)
def test_missing_fidelity(monkeypatch, config_file):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with pytest.raises(RuntimeError) as exc:
        orion.core.cli.main(["hunt", "--config", config_file,
                             "./black_box.py", "-x~uniform(-50, 50)"])
    assert "https://orion.readthedocs.io/en/develop/user/algorithms.html" in str(exc.value)


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
@pytest.mark.parametrize('config_file', config_files)
def test_simple(monkeypatch, config_file):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["hunt", "--config", config_file,
                         "./black_box.py", "-x~uniform(-50, 50)"])

    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)

    storage = get_storage()
    exp = list(storage.fetch_experiments({'name': config['name']}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    assert exp['name'] == config['name']
    assert exp['pool_size'] == 1
    assert exp['max_trials'] == 100
    assert exp['algorithms'] == config['algorithms']
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'orion_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~uniform(-50, 50)']

    trials = storage.fetch_trials(uid=exp_id)
    assert len(trials) <= config['max_trials']
    assert trials[-1].status == 'completed'

    best_trial = next(iter(sorted(trials, key=lambda trial: trial.objective.value)))
    assert best_trial.objective.name == 'example_objective'
    assert abs(best_trial.objective.value - 23.4) < 1e-5
    assert len(best_trial.params) == 1
    param = best_trial.params[0]
    assert param.name == '/x'
    assert param.type == 'real'


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
@pytest.mark.parametrize('config_file', fidelity_config_files)
def test_with_fidelity(database, monkeypatch, config_file):
    """Test a scenario with fidelity."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["hunt", "--config", config_file,
                         "./black_box.py", "-x~uniform(-50, 50)", "--fidelity~fidelity(1,10,4)"])

    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)

    storage = get_storage()
    exp = list(storage.fetch_experiments({'name': config['name']}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    assert exp['name'] == config['name']
    assert exp['pool_size'] == 1
    assert exp['max_trials'] == 100
    assert exp['algorithms'] == config['algorithms']
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'orion_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~uniform(-50, 50)', "--fidelity~fidelity(1,10,4)"]

    trials = storage.fetch_trials(uid=exp_id)
    assert len(trials) <= config['max_trials']
    assert trials[-1].status == 'completed'

    best_trial = next(iter(sorted(trials, key=lambda trial: trial.objective.value)))
    assert best_trial.objective.name == 'example_objective'
    assert abs(best_trial.objective.value - 23.4) < 1e-5
    assert len(best_trial.params) == 2
    fidelity = best_trial.params[0]
    assert fidelity.name == '/fidelity'
    assert fidelity.type == 'fidelity'
    assert fidelity.value == 10
    param = best_trial.params[1]
    assert param.name == '/x'
    assert param.type == 'real'
