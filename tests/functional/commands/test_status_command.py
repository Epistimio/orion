#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the status command."""
import os

import pytest

import orion.core.cli


def test_no_experiments(clean_db, monkeypatch, capsys):
    """Test status with no experiments."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    assert captured == "No experiment found\n"


def test_no_version_backward_compatible(clean_db, one_experiment_no_version, capsys):
    """Test status with no experiments."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-no-version-v1
=============================
empty


"""
    assert captured == expected


def test_python_api(clean_db, with_experiment_using_python_api, capsys):
    """Test status with experiments built using python api."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
empty


from-python-api-v1
==================
empty


"""
    assert captured == expected


def test_experiment_without_trials_wout_ac(clean_db, one_experiment, capsys):
    """Test status with only one experiment and no trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
empty


"""
    assert captured == expected


def test_experiment_wout_success_wout_ac(clean_db, single_without_success, capsys):
    """Test status with only one experiment and no successful trial."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
status         quantity
-----------  ----------
broken                1
interrupted           1
new                   1
reserved              1
suspended             1


"""
    assert captured == expected


def test_experiment_number_same_list_status(clean_db,
                                            single_without_success, capsys):
    """Test status and list command output the consistent number of experiments"""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
status         quantity
-----------  ----------
broken                1
interrupted           1
new                   1
reserved              1
suspended             1


"""
    assert captured == expected

    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp-v1\n"


def test_experiment_w_trials_wout_ac(clean_db, single_with_trials, capsys):
    """Test status with only one experiment and all trials."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
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
test_double_exp-v1
==================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


test_single_exp-v1
==================
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
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
test_double_exp-v1
==================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child-v1
  ========================
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
test_double_exp-v1
==================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child-v1
  ========================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


test_single_exp-v1
==================
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
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
test_double_exp-v1
==================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child-v1
  ========================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


  test_double_exp_child2-v1
  =========================
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
test_double_exp-v1
==================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child-v1
  ========================
  status         quantity
  -----------  ----------
  broken                1
  completed             1
  interrupted           1
  new                   1
  reserved              1
  suspended             1


    test_double_exp_grand_child-v1
    ==============================
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
test_single_exp-v1
==================
id     status    best objective
-----  --------  ----------------
empty


"""

    assert captured == expected


def test_one_w_trials_w_a_wout_c(clean_db, single_with_trials, capsys):
    """Test experiment, with all trials, with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken
47564e5e390348b9d1335d4013895eb4  completed            0
aefd38473f108016fd4842aa855732ff  interrupted
0695f63ecaf7d78f4b85d4cb344e0dc0  new
b0ea9850c09370215b45b81edd33c7d3  reserved
b49e902aebccce14e834d96e411f896e  suspended


"""

    assert captured == expected


def test_one_wout_success_w_a_wout_c(clean_db, single_without_success, capsys):
    """Test experiment, without success, with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
id                                status
--------------------------------  -----------
9f360d1b4eb2707f19dd619d0d898dd9  broken
aefd38473f108016fd4842aa855732ff  interrupted
0695f63ecaf7d78f4b85d4cb344e0dc0  new
b0ea9850c09370215b45b81edd33c7d3  reserved
b49e902aebccce14e834d96e411f896e  suspended


"""

    assert captured == expected


def test_two_unrelated_w_a_wout_c(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken
47564e5e390348b9d1335d4013895eb4  completed            0
aefd38473f108016fd4842aa855732ff  interrupted
0695f63ecaf7d78f4b85d4cb344e0dc0  new
b0ea9850c09370215b45b81edd33c7d3  reserved
b49e902aebccce14e834d96e411f896e  suspended


"""

    assert captured == expected


def test_two_related_w_a_wout_c(clean_db, family_with_trials, capsys):
    """Test two related experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  9bd1ebc475bcb9e077a9e81a7c954a65  broken
  3c1af2af2c8dc9862df2cef0a65d6e1f  completed
  614ec3fc127d52129bc9d66d9aeec36c  interrupted
  4487e7fc87c288d254f94dfa82cd79cc  new
  7877287c718d7844570003fd654f66ba  reserved
  ff997e666e20c5a8c1a816dde0b5e2e9  suspended


"""

    assert captured == expected


def test_three_unrelated_w_a_wout_c(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  9bd1ebc475bcb9e077a9e81a7c954a65  broken
  3c1af2af2c8dc9862df2cef0a65d6e1f  completed
  614ec3fc127d52129bc9d66d9aeec36c  interrupted
  4487e7fc87c288d254f94dfa82cd79cc  new
  7877287c718d7844570003fd654f66ba  reserved
  ff997e666e20c5a8c1a816dde0b5e2e9  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken
47564e5e390348b9d1335d4013895eb4  completed            0
aefd38473f108016fd4842aa855732ff  interrupted
0695f63ecaf7d78f4b85d4cb344e0dc0  new
b0ea9850c09370215b45b81edd33c7d3  reserved
b49e902aebccce14e834d96e411f896e  suspended


"""

    assert captured == expected


def test_three_related_w_a_wout_c(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  9bd1ebc475bcb9e077a9e81a7c954a65  broken
  3c1af2af2c8dc9862df2cef0a65d6e1f  completed
  614ec3fc127d52129bc9d66d9aeec36c  interrupted
  4487e7fc87c288d254f94dfa82cd79cc  new
  7877287c718d7844570003fd654f66ba  reserved
  ff997e666e20c5a8c1a816dde0b5e2e9  suspended


  test_double_exp_child2-v1
  =========================
  id                                status
  --------------------------------  -----------
  2c2b64df1859b45a0b01362ca146584a  broken
  225b4a17dd29d5c0423a81c1ddda8f0e  completed
  fb0bb45bd0a45225e2368a8158df0427  interrupted
  57bd3071c7c1c39ceb997a7b37c5470d  new
  673449a3910fdea777ac8cb8576cdbe3  reserved
  01a38cce74701c3b40eb3d92143bc90f  suspended


"""

    assert captured == expected


def test_three_related_branch_w_a_wout_c(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments in a branch with --all."""
    orion.core.cli.main(['status', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  9bd1ebc475bcb9e077a9e81a7c954a65  broken
  3c1af2af2c8dc9862df2cef0a65d6e1f  completed
  614ec3fc127d52129bc9d66d9aeec36c  interrupted
  4487e7fc87c288d254f94dfa82cd79cc  new
  7877287c718d7844570003fd654f66ba  reserved
  ff997e666e20c5a8c1a816dde0b5e2e9  suspended


    test_double_exp_grand_child-v1
    ==============================
    id                                status
    --------------------------------  -----------
    82f82a325b7cf09251a34c9264e1812a  broken
    94baf74a4e94f800b6865d8ab5675428  completed
    e24b2e542c0869064abdb20c2de250eb  interrupted
    960bad983c3ee6349b8767fe452ecbb3  new
    8d8578e31c740c1c0fc385c961702481  reserved
    584375e2b32af0573f4692cb47a2ec99  suspended


"""

    assert captured == expected


def test_two_unrelated_w_c_wout_a(clean_db, unrelated_with_trials, capsys):
    """Test two unrelated experiments with --collapse."""
    orion.core.cli.main(['status', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


test_single_exp-v1
==================
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
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
test_double_exp-v1
==================
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
test_double_exp-v1
==================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   2
reserved              1
suspended             1


test_single_exp-v1
==================
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
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
test_double_exp-v1
==================
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
test_double_exp-v1
==================
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
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken
47564e5e390348b9d1335d4013895eb4  completed            0
aefd38473f108016fd4842aa855732ff  interrupted
0695f63ecaf7d78f4b85d4cb344e0dc0  new
b0ea9850c09370215b45b81edd33c7d3  reserved
b49e902aebccce14e834d96e411f896e  suspended


"""

    assert captured == expected


def test_two_related_w_ac(clean_db, family_with_trials, capsys):
    """Test two related experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
d5f1c1cae188608b581ded20cd198679  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


"""

    assert captured == expected


def test_three_unrelated_w_ac(clean_db, three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
d5f1c1cae188608b581ded20cd198679  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9f360d1b4eb2707f19dd619d0d898dd9  broken
47564e5e390348b9d1335d4013895eb4  completed            0
aefd38473f108016fd4842aa855732ff  interrupted
0695f63ecaf7d78f4b85d4cb344e0dc0  new
b0ea9850c09370215b45b81edd33c7d3  reserved
b49e902aebccce14e834d96e411f896e  suspended


"""

    assert captured == expected


def test_three_related_w_ac(clean_db, three_family_with_trials, capsys):
    """Test three related experiments with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
d5f1c1cae188608b581ded20cd198679  new
e5bf1dd6dec1a0c690ed62ff9146e5b8  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


"""

    assert captured == expected


def test_three_related_branch_w_ac(clean_db, three_family_branch_with_trials, capsys):
    """Test three related experiments in a branch with --collapse and --all."""
    orion.core.cli.main(['status', '--collapse', '--all'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
c2187f4954884c801e423d851aec9a0b  broken
e42cc22a15188d72df315b9eac79c9c0  completed
b849f69cc3a77f39382d7435d0d41b14  interrupted
7fbbd152f7ca2c064bf00441e311609d  new
d5f1c1cae188608b581ded20cd198679  new
183148e187a1399989a06ffb02059920  new
667513aa2cb2244bee9c4f41c7ff1cea  reserved
557b9fdb9f96569dff7eb2de10d3946f  suspended


"""

    assert captured == expected


def test_no_experiments_w_name(clean_db, monkeypatch, capsys):
    """Test status when --name <exp> does not exist."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['status', '--name', 'test_ghost_exp'])

    captured = capsys.readouterr().out

    assert captured == "No experiment found\n"


def test_experiment_wout_child_w_name(clean_db, unrelated_with_trials, capsys):
    """Test status with the name argument and no child."""
    orion.core.cli.main(['status', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
status         quantity    min obj
-----------  ----------  ---------
broken                1
completed             1          0
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
test_double_exp-v1
==================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   1
reserved              1
suspended             1


  test_double_exp_child-v1
  ========================
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


def test_experiment_w_parent_w_name(clean_db, three_experiments_with_trials, capsys):
    """Test status with the name argument and one parent."""
    orion.core.cli.main(['status', '--name', 'test_double_exp_child'])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp_child-v1
========================
status         quantity
-----------  ----------
broken                1
completed             1
interrupted           1
new                   2
reserved              2
suspended             1


"""

    assert captured == expected


def test_experiment_same_name_wout_exv(clean_db, three_experiments_same_name, capsys):
    """Test status with three experiments having the same name but different versions."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v3
==================
empty


"""

    assert captured == expected


def test_experiment_same_name_wout_exv_w_name(clean_db, three_experiments_same_name, capsys):
    """Test status with three experiments having the same name but different versions."""
    orion.core.cli.main(['status', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v3
==================
empty


"""

    assert captured == expected


def test_experiment_same_name_wout_exv_w_child_w_name(clean_db,
                                                      three_experiments_family_same_name, capsys):
    """Test status name with two experiments having the same name and one with a child."""
    orion.core.cli.main(['status', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
empty


  test_single_exp-v2
  ==================
  empty


  test_single_exp_child-v1
  ========================
  empty


"""

    assert captured == expected


def test_experiment_same_name_wout_exv_w_c_w_child_w_name(
        clean_db, three_experiments_family_same_name, capsys):
    """Test status name collapsed with two experiments having the same name and one with a child."""
    orion.core.cli.main(['status', '--name', 'test_single_exp', '--collapse'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v2
==================
empty


"""

    assert captured == expected


def test_experiment_same_name_wout_exv_w_child(clean_db,
                                               three_experiments_family_same_name, capsys):
    """Test status with two experiments having the same name and one with a child."""
    orion.core.cli.main(['status'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
empty


  test_single_exp-v2
  ==================
  empty


  test_single_exp_child-v1
  ========================
  empty


"""

    assert captured == expected


def test_experiment_same_name_w_exv(clean_db, three_experiments_same_name, capsys):
    """Test status with three experiments with the same name and `--expand-verions`."""
    orion.core.cli.main(['status', '--expand-versions'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
empty


  test_single_exp-v2
  ==================
  empty


    test_single_exp-v3
    ==================
    empty


"""

    assert captured == expected


def test_experiment_same_name_w_exv_w_child(clean_db, three_experiments_family_same_name, capsys):
    """Test status with two experiments having the same name and one with a child."""
    orion.core.cli.main(['status', '--expand-versions'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
empty


  test_single_exp-v2
  ==================
  empty


  test_single_exp_child-v1
  ========================
  empty


"""

    assert captured == expected


def test_experiment_specific_version(clean_db, three_experiments_same_name, capsys):
    """Test status using `--version`."""
    orion.core.cli.main(['status', '--version', '2'])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v2
==================
empty


"""

    assert captured == expected


def test_experiment_cant_use_version(clean_db, three_experiments_same_name):
    """Test status using `--version`."""
    with pytest.raises(RuntimeError) as ex:
        orion.core.cli.main(['status', '--version', '2', '--collapse'])

    assert 'collapse' in str(ex.value)

    with pytest.raises(RuntimeError) as ex:
        orion.core.cli.main(['status', '--version', '2', '--expand-versions'])

    assert 'expand-versions' in str(ex.value)
