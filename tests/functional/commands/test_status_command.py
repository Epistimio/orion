#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the status command."""
import os

import orion.core.cli


def test_no_experiments(clean_db, monkeypatch, capsys):
    """Test status with no experiments."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    assert captured == ""


def test_experiment_without_trials_wout_ac(clean_db, one_experiment, capsys):
    """Test status with only one experiment and no trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp - v.1
=====================
empty


"""
    assert captured == expected


def test_experiment_wout_success_wout_ac(clean_db, single_without_success, capsys):
    """Test status with only one experiment and no successful trial."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
interrupted           1
new                   1
reserved              1
suspended             1


"""
    assert captured == expected


def test_experiment_w_trials_wout_ac(clean_db, single_with_trials, capsys):
    """Test status with only one experiment and all trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp - v.1
=====================
status         quantity  params      min obj
-----------  ----------  --------  ---------
broken                1
completed             1  x=100             0
interrupted           1
new                   1
reserved              1
suspended             1


"""
    assert captured == expected


def test_two_unrelated_w_trials_wout_ac(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments, with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


test_single_exp - v.1
=====================
status         quantity  params      min obj
-----------  ----------  --------  ---------
broken                1
completed             1  x=100             0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_two_related_w_trials_wout_ac(clean_db, family_with_trials, capsys):
    """Test two related experiments, with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child - v.1
  ===========================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


"""

    assert captured == expected


def test_three_unrelated_wout_ac(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child - v.1
  ===========================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


test_single_exp - v.1
=====================
status         quantity  params      min obj
-----------  ----------  --------  ---------
broken                1
completed             1  x=100             0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_three_related_wout_ac(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child - v.1
  ===========================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


  test_double_exp_child2 - v.1
  ============================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


"""

    assert captured == expected


def test_three_related_branch_wout_ac(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments with all types of trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child - v.1
  ===========================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


    test_double_exp_grand_child - v.1
    =================================
    status         quantity
    -----------  ----------
    broken                1
    completed             1
    interrupted           1
    new                   1
    reserved              1
    suspended             1


"""

    assert captured == expected


def test_one_wout_trials_w_a_wout_c(clean_db, one_experiment, capsys):
    """Test experiments, without trials, with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp - v.1
=====================
id    status    params    best objective
----  --------  --------  ----------------


"""

    assert captured == expected


def test_one_w_trials_w_a_wout_c(clean_db, single_with_trials, capsys):
    """Test experiment, with all trials, with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp - v.1
=====================
id                                status       params      min obj
--------------------------------  -----------  --------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken       x=4
47564e5e390348b9d1335d4013895eb4  completed    x=100             0
aefd38473f108016fd4842aa855732ff  interrupted  x=3
0695f63ecaf7d78f4b85d4cb344e0dc0  new          x=0
b0ea9850c09370215b45b81edd33c7d3  reserved     x=1
b49e902aebccce14e834d96e411f896e  suspended    x=2


"""

    assert captured == expected


def test_one_wout_success_w_a_wout_c(clean_db, single_without_success, capsys):
    """Test experiment, without success, with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
9f360d1b4eb2707f19dd619d0d898dd9  broken       x=4
aefd38473f108016fd4842aa855732ff  interrupted  x=3
0695f63ecaf7d78f4b85d4cb344e0dc0  new          x=0
b0ea9850c09370215b45b81edd33c7d3  reserved     x=1
b49e902aebccce14e834d96e411f896e  suspended    x=2


"""

    assert captured == expected


def test_two_unrelated_w_a_wout_c(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


test_single_exp - v.1
=====================
id                                status       params      min obj
--------------------------------  -----------  --------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken       x=4
47564e5e390348b9d1335d4013895eb4  completed    x=100             0
aefd38473f108016fd4842aa855732ff  interrupted  x=3
0695f63ecaf7d78f4b85d4cb344e0dc0  new          x=0
b0ea9850c09370215b45b81edd33c7d3  reserved     x=1
b49e902aebccce14e834d96e411f896e  suspended    x=2


"""

    assert captured == expected


def test_two_related_w_a_wout_c(clean_db, family_with_trials, capsys):
    """Test two related experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


  test_double_exp_child - v.1
  ===========================
  id                                status       params
  --------------------------------  -----------  --------
  9bd1ebc475bcb9e077a9e81a7c954a65  broken       x=5, y=5
  3c1af2af2c8dc9862df2cef0a65d6e1f  completed    x=3, y=3
  614ec3fc127d52129bc9d66d9aeec36c  interrupted  x=4, y=4
  4487e7fc87c288d254f94dfa82cd79cc  new          x=0, y=0
  7877287c718d7844570003fd654f66ba  reserved     x=1, y=1
  ff997e666e20c5a8c1a816dde0b5e2e9  suspended    x=2, y=2


"""

    assert captured == expected


def test_three_unrelated_w_a_wout_c(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


  test_double_exp_child - v.1
  ===========================
  id                                status       params
  --------------------------------  -----------  --------
  9bd1ebc475bcb9e077a9e81a7c954a65  broken       x=5, y=5
  3c1af2af2c8dc9862df2cef0a65d6e1f  completed    x=3, y=3
  614ec3fc127d52129bc9d66d9aeec36c  interrupted  x=4, y=4
  4487e7fc87c288d254f94dfa82cd79cc  new          x=0, y=0
  7877287c718d7844570003fd654f66ba  reserved     x=1, y=1
  ff997e666e20c5a8c1a816dde0b5e2e9  suspended    x=2, y=2


test_single_exp - v.1
=====================
id                                status       params      min obj
--------------------------------  -----------  --------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken       x=4
47564e5e390348b9d1335d4013895eb4  completed    x=100             0
aefd38473f108016fd4842aa855732ff  interrupted  x=3
0695f63ecaf7d78f4b85d4cb344e0dc0  new          x=0
b0ea9850c09370215b45b81edd33c7d3  reserved     x=1
b49e902aebccce14e834d96e411f896e  suspended    x=2


"""

    assert captured == expected


def test_three_related_w_a_wout_c(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


  test_double_exp_child - v.1
  ===========================
  id                                status       params
  --------------------------------  -----------  --------
  9bd1ebc475bcb9e077a9e81a7c954a65  broken       x=5, y=5
  3c1af2af2c8dc9862df2cef0a65d6e1f  completed    x=3, y=3
  614ec3fc127d52129bc9d66d9aeec36c  interrupted  x=4, y=4
  4487e7fc87c288d254f94dfa82cd79cc  new          x=0, y=0
  7877287c718d7844570003fd654f66ba  reserved     x=1, y=1
  ff997e666e20c5a8c1a816dde0b5e2e9  suspended    x=2, y=2


  test_double_exp_child2 - v.1
  ============================
  id                                status       params
  --------------------------------  -----------  ----------
  2c2b64df1859b45a0b01362ca146584a  broken       x=5, z=500
  225b4a17dd29d5c0423a81c1ddda8f0e  completed    x=3, z=300
  fb0bb45bd0a45225e2368a8158df0427  interrupted  x=4, z=400
  57bd3071c7c1c39ceb997a7b37c5470d  new          x=0, z=0
  673449a3910fdea777ac8cb8576cdbe3  reserved     x=1, z=100
  01a38cce74701c3b40eb3d92143bc90f  suspended    x=2, z=200


"""

    assert captured == expected


def test_three_related_branch_w_a_wout_c(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments in a branch with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


  test_double_exp_child - v.1
  ===========================
  id                                status       params
  --------------------------------  -----------  --------
  9bd1ebc475bcb9e077a9e81a7c954a65  broken       x=5, y=5
  3c1af2af2c8dc9862df2cef0a65d6e1f  completed    x=3, y=3
  614ec3fc127d52129bc9d66d9aeec36c  interrupted  x=4, y=4
  4487e7fc87c288d254f94dfa82cd79cc  new          x=0, y=0
  7877287c718d7844570003fd654f66ba  reserved     x=1, y=1
  ff997e666e20c5a8c1a816dde0b5e2e9  suspended    x=2, y=2


    test_double_exp_grand_child - v.1
    =================================
    id                                status       params
    --------------------------------  -----------  ----------------
    82f82a325b7cf09251a34c9264e1812a  broken       x=5, y=50, z=500
    94baf74a4e94f800b6865d8ab5675428  completed    x=3, y=30, z=300
    e24b2e542c0869064abdb20c2de250eb  interrupted  x=4, y=40, z=400
    960bad983c3ee6349b8767fe452ecbb3  new          x=0, y=0, z=0
    8d8578e31c740c1c0fc385c961702481  reserved     x=1, y=10, z=100
    584375e2b32af0573f4692cb47a2ec99  suspended    x=2, y=20, z=200


"""

    assert captured == expected


def test_two_unrelated_w_c_wout_a(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


test_single_exp - v.1
=====================
status         quantity  params      min obj
-----------  ----------  --------  ---------
broken                1
completed             1  x=100             0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_two_related_w_c_wout_a(clean_db, family_with_trials, capsys):
    """Test two related experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   2
reserved              1
suspended             1


"""

    assert captured == expected


def test_three_unrelated_w_c_wout_a(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   2
reserved              1
suspended             1


test_single_exp - v.1
=====================
status         quantity  params      min obj
-----------  ----------  --------  ---------
broken                1
completed             1  x=100             0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_three_related_w_c_wout_a(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   3
reserved              1
suspended             1


"""

    assert captured == expected


def test_three_related_branch_w_c_wout_a(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   3
reserved              1
suspended             1


"""

    assert captured == expected


def test_two_unrelated_w_ac(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


test_single_exp - v.1
=====================
id                                status       params      min obj
--------------------------------  -----------  --------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken       x=4
47564e5e390348b9d1335d4013895eb4  completed    x=100             0
aefd38473f108016fd4842aa855732ff  interrupted  x=3
0695f63ecaf7d78f4b85d4cb344e0dc0  new          x=0
b0ea9850c09370215b45b81edd33c7d3  reserved     x=1
b49e902aebccce14e834d96e411f896e  suspended    x=2


"""

    assert captured == expected


def test_two_related_w_ac(clean_db, family_with_trials, capsys):
    """Test two related experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
d5f1c1cae188608b581ded20cd198679  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


"""

    assert captured == expected


def test_three_unrelated_w_ac(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
d5f1c1cae188608b581ded20cd198679  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


test_single_exp - v.1
=====================
id                                status       params      min obj
--------------------------------  -----------  --------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken       x=4
47564e5e390348b9d1335d4013895eb4  completed    x=100             0
aefd38473f108016fd4842aa855732ff  interrupted  x=3
0695f63ecaf7d78f4b85d4cb344e0dc0  new          x=0
b0ea9850c09370215b45b81edd33c7d3  reserved     x=1
b49e902aebccce14e834d96e411f896e  suspended    x=2


"""

    assert captured == expected


def test_three_related_w_ac(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
d5f1c1cae188608b581ded20cd198679  new          x=0
e5bf1dd6dec1a0c690ed62ff9146e5b8  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


"""

    assert captured == expected


def test_three_related_branch_w_ac(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments in a branch with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
id                                status       params
--------------------------------  -----------  --------
c2187f4954884c801e423d851aec9a0b  broken       x=5
e42cc22a15188d72df315b9eac79c9c0  completed    x=3
b849f69cc3a77f39382d7435d0d41b14  interrupted  x=4
7fbbd152f7ca2c064bf00441e311609d  new          x=0
d5f1c1cae188608b581ded20cd198679  new          x=0
183148e187a1399989a06ffb02059920  new          x=0
667513aa2cb2244bee9c4f41c7ff1cea  reserved     x=1
557b9fdb9f96569dff7eb2de10d3946f  suspended    x=2


"""

    assert captured == expected


def test_experiment_wout_child_w_name(clean_db, unrelated_with_trials, capsys):
    """Test status with the name argument and no child."""
    orion.core.cli.main(['status', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp - v.1
=====================
status         quantity  params      min obj
-----------  ----------  --------  ---------
broken                1
completed             1  x=100             0
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected


def test_experiment_w_child_w_name(clean_db, three_experiments_with_trials, capsys):
    """Test status with the name argument and one child."""
    orion.core.cli.main(['status', '--name', 'test_double_exp'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp - v.1
=====================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


"""

    assert captured == expected
