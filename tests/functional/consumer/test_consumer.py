#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test against possible scenarios that `orion.core.worker.consumer` may
have to handle.
"""
import os
import signal
import subprocess
import time

import pytest


@pytest.mark.usefixtures("clean_db")
def test_consumer_exit_success(database, monkeypatch):
    """Test a scenario in which the black box works fine."""
    EXP_NAME = "consumer_exit_success"
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=1", "--max-trials=10",
                                "./success_box.py", "-x~norm(34, 3)"])
    rcode = process.wait()
    assert rcode == 0

    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 1
    assert exp['max_trials'] == 10
    assert exp['status'] == 'done'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 10
    assert len(ntrials) == 0
    assert len(rtrials) == 0
    assert len(strials) == 0
    assert len(ctrials) == 10
    assert len(itrials) == 0
    assert len(btrials) == 0


@pytest.mark.usefixtures("clean_db")
def test_consumer_exit_fail(database, monkeypatch, capfd):
    """Test a scenario in which the black box has a bug."""
    EXP_NAME = "consumer_exit_fail"
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=5", "--max-trials=10",
                                "./fail_box.py", "-x~norm(34, 3)"])
    rcode = process.wait()
    assert rcode == 0

    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 5
    assert exp['max_trials'] == 10
    assert exp['status'] == 'broken'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 5
    assert len(ntrials) == 2
    assert len(rtrials) == 0
    assert len(strials) == 0
    assert len(ctrials) == 0
    assert len(itrials) == 0
    assert len(btrials) == 3
    capfd.readouterr()  # Suppress fd level 1 & 2


@pytest.mark.usefixtures("clean_db")
def test_consumer_exit_2(database, monkeypatch):
    """Test a scenario in which the black box exits because of cmd bad options."""
    EXP_NAME = "consumer_exit_2"
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=3", "--max-trials=10",
                                "./success_box.py", "-y~norm(34, 3)"])
    rcode = process.wait()
    assert rcode == 2

    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 3
    assert exp['max_trials'] == 10
    assert exp['status'] == 'broken'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 3
    assert len(ntrials) == 2
    assert len(rtrials) == 0
    assert len(strials) == 0
    assert len(ctrials) == 0
    assert len(itrials) == 0
    assert len(btrials) == 1


@pytest.mark.skip("Cannot deal with asychronicity deterministically.")
@pytest.mark.usefixtures("clean_db")
def test_consumer_killed_with_sigint(database, monkeypatch, capfd):
    """Test a scenario in which the black box works fine, until a SIGINT arrives."""
    EXP_NAME = "consumer_killed_with_sigint"
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=3", "--max-trials=6",
                                "./success_box.py", "-x~norm(34, 3)"])

    time.sleep(3)
    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    cc = 0
    while cc < 1:
        cc = database.trials.count({'experiment': exp_id, 'status': 'completed'})
    time.sleep(0.5)
    process.send_signal(signal.SIGINT)

    rcode = process.wait()
    assert rcode == 1
    capfd.readouterr()  # Suppress fd level 1 & 2

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 3
    assert exp['max_trials'] == 6
    # XXX: Is this the experiment's status we are expecting in this scenario?
    assert exp['status'] == 'pending'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 3
    assert len(ntrials) == 1
    assert len(rtrials) == 0
    assert len(strials) == 0
    assert len(ctrials) == 1
    assert len(itrials) == 1
    assert len(btrials) == 0

    interrupted_trial_id = itrials[0]['_id']

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=3", "--max-trials=6",
                                "./success_box.py", "-x~norm(34, 3)"])
    rcode = process.wait()
    assert rcode == 0

    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 3
    assert exp['max_trials'] == 6
    assert exp['status'] == 'done'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 6
    assert len(ntrials) == 0
    assert len(rtrials) == 0
    assert len(strials) == 0
    assert len(ctrials) == 6
    assert len(itrials) == 0
    assert len(btrials) == 0

    assert any(trial['_id'] == interrupted_trial_id for trial in ctrials)


@pytest.mark.skip("Cannot deal with asychronicity deterministically.")
@pytest.mark.usefixtures("clean_db")
def test_consumer_killed_with_sigterm(database, monkeypatch):
    """Test a scenario in which the black box works fine, until a SIGTERM arrives.

    A trial remains under the status 'reserved' and it is never resumed.
    """
    EXP_NAME = "consumer_killed_with_sigterm"
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=3", "--max-trials=6",
                                "./success_box.py", "-x~norm(34, 3)"])

    time.sleep(3)
    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    cc = 0
    while cc < 1:
        cc = database.trials.count({'experiment': exp_id, 'status': 'completed'})
    time.sleep(0.5)
    process.send_signal(signal.SIGTERM)

    rcode = process.wait()
    assert rcode == -15

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 3
    assert exp['max_trials'] == 6
    # XXX: Is this the experiment's status we are expecting in this scenario?
    assert exp['status'] == 'pending'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 3
    assert len(ntrials) == 1
    assert len(rtrials) == 1
    assert len(strials) == 0
    assert len(ctrials) == 1
    assert len(itrials) == 0
    assert len(btrials) == 0

    reserved_trial_id = rtrials[0]['_id']

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=3", "--max-trials=6",
                                "./success_box.py", "-x~norm(34, 3)"])
    rcode = process.wait()
    assert rcode == 0

    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 3
    assert exp['max_trials'] == 6
    assert exp['status'] == 'done'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 9
    assert len(ntrials) == 0
    assert len(rtrials) == 1
    assert len(strials) == 0
    assert len(ctrials) == 8
    assert len(itrials) == 0
    assert len(btrials) == 0

    assert all(trial['_id'] != reserved_trial_id for trial in ctrials)
    assert rtrials[0]['_id'] == reserved_trial_id


@pytest.mark.skip("Cannot deal with asychronicity deterministically.")
@pytest.mark.usefixtures("clean_db")
def test_consumer_killed_with_sigkill(database, monkeypatch):
    """Test a scenario in which the black box works fine, until a SIGKILL arrives.

    A trial remains under the status 'reserved' and it is never resumed.
    """
    EXP_NAME = "consumer_killed_with_sigkill"
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=3", "--max-trials=6",
                                "./success_box.py", "-x~norm(34, 3)"])

    time.sleep(3)
    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    cc = 0
    while cc < 1:
        cc = database.trials.count({'experiment': exp_id, 'status': 'completed'})
    time.sleep(0.5)
    process.send_signal(signal.SIGKILL)

    rcode = process.wait()
    assert rcode == -9

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 3
    assert exp['max_trials'] == 6
    # XXX: Is this the experiment's status we are expecting in this scenario?
    assert exp['status'] == 'pending'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 3
    assert len(ntrials) == 1
    assert len(rtrials) == 1
    assert len(strials) == 0
    assert len(ctrials) == 1
    assert len(itrials) == 0
    assert len(btrials) == 0

    reserved_trial_id = rtrials[0]['_id']

    process = subprocess.Popen(["orion", "hunt", "-n", EXP_NAME,
                                "--pool-size=3", "--max-trials=6",
                                "./success_box.py", "-x~norm(34, 3)"])
    rcode = process.wait()
    assert rcode == 0

    exp = list(database.experiments.find({'name': EXP_NAME}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    assert exp['name'] == EXP_NAME
    assert exp['pool_size'] == 3
    assert exp['max_trials'] == 6
    assert exp['status'] == 'done'

    trials = list(database.trials.find({'experiment': exp_id}))
    ntrials = list(database.trials.find({'experiment': exp_id, 'status': 'new'}))
    rtrials = list(database.trials.find({'experiment': exp_id, 'status': 'reserved'}))
    strials = list(database.trials.find({'experiment': exp_id, 'status': 'suspended'}))
    ctrials = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    itrials = list(database.trials.find({'experiment': exp_id, 'status': 'interrupted'}))
    btrials = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))

    assert len(trials) == 9
    assert len(ntrials) == 0
    assert len(rtrials) == 1
    assert len(strials) == 0
    assert len(ctrials) == 8
    assert len(itrials) == 0
    assert len(btrials) == 0

    assert all(trial['_id'] != reserved_trial_id for trial in ctrials)
    assert rtrials[0]['_id'] == reserved_trial_id
