#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for demo purposes."""
import os
import subprocess

import numpy
import pytest


@pytest.mark.usefixtures("clean_db")
def test_demo(database):
    """Test a simple usage scenario."""
    curdir = os.path.abspath(os.path.curdir)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    process = subprocess.Popen(["mopt", "--config", "./moptconfig.yaml",
                                "./black_box.py", "-x~random(-50, 50)"])
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
    assert 'mopt_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~random(-50, 50)']

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

    os.chdir(curdir)
