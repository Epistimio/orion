#!/usr/bin/env python
"""Perform a functional test of the status command."""
import os

import pytest

import orion.core.cli


def test_no_experiments(orionstate, monkeypatch, capsys):
    """Test status with no experiments."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["status"])

    captured = capsys.readouterr().out

    assert captured == "No experiment found\n"


def test_no_version_backward_compatible(one_experiment_no_version, capsys, storage):
    """Test status with no experiments."""
    orion.core.cli.main(["status"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-no-version-v1
=============================
empty


"""
    assert captured == expected


def test_python_api(with_experiment_using_python_api, capsys):
    """Test status with experiments built using python api."""
    orion.core.cli.main(["status"])

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


def test_missing_conf_file(with_experiment_missing_conf_file, capsys):
    """Test status can handle experiments when the user script config file is missing"""
    orion.core.cli.main(["status"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
empty


"""
    assert captured == expected


def test_experiment_without_trials_wout_ac(one_experiment, capsys):
    """Test status with only one experiment and no trials."""
    orion.core.cli.main(["status"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
empty


"""
    assert captured == expected


def test_experiment_wout_success_wout_ac(single_without_success, capsys):
    """Test status with only one experiment and no successful trial."""
    orion.core.cli.main(["status"])

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


def test_experiment_number_same_list_status(single_without_success, capsys):
    """Test status and list command output the consistent number of experiments"""
    orion.core.cli.main(["status"])

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

    orion.core.cli.main(["list"])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp-v1\n"


def test_experiment_w_trials_wout_ac(single_with_trials, capsys):
    """Test status with only one experiment and all trials."""
    orion.core.cli.main(["status"])

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


def test_two_unrelated_w_trials_wout_ac(unrelated_with_trials, capsys):
    """Test two unrelated experiments, with all types of trials."""
    orion.core.cli.main(["status"])

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


def test_two_related_w_trials_wout_ac(family_with_trials, capsys):
    """Test two related experiments, with all types of trials."""
    orion.core.cli.main(["status"])

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


def test_three_unrelated_wout_ac(three_experiments_with_trials, capsys):
    """Test three unrelated experiments with all types of trials."""
    orion.core.cli.main(["status"])

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


def test_three_related_wout_ac(three_family_with_trials, capsys):
    """Test three related experiments with all types of trials."""
    orion.core.cli.main(["status"])

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


def test_three_related_branch_wout_ac(three_family_branch_with_trials, capsys):
    """Test three related experiments with all types of trials."""
    orion.core.cli.main(["status"])

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


def test_one_wout_trials_w_a_wout_c(one_experiment, capsys):
    """Test experiments, without trials, with --all."""
    orion.core.cli.main(["status", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
id     status    best objective
-----  --------  ----------------
empty


"""

    assert captured == expected


def test_one_w_trials_w_a_wout_c(single_with_trials, capsys):
    """Test experiment, with all trials, with --all."""
    orion.core.cli.main(["status", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9a351eb75be219e27327e16dcb2a7b3b  broken
43b457cde7502ccb584f724d4562ed27  completed            0
156d55adbfc36923421cb05da1116a34  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_one_wout_success_w_a_wout_c(single_without_success, capsys):
    """Test experiment, without success, with --all."""
    orion.core.cli.main(["status", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v1
==================
id                                status
--------------------------------  -----------
9a351eb75be219e27327e16dcb2a7b3b  broken
156d55adbfc36923421cb05da1116a34  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_two_unrelated_w_a_wout_c(unrelated_with_trials, capsys):
    """Test two unrelated experiments with --all."""
    orion.core.cli.main(["status", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9a351eb75be219e27327e16dcb2a7b3b  broken
43b457cde7502ccb584f724d4562ed27  completed            0
156d55adbfc36923421cb05da1116a34  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_two_related_w_a_wout_c(family_with_trials, capsys):
    """Test two related experiments with --all."""
    orion.core.cli.main(["status", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  be3832db95702cd617eb4215d9eaa7dc  broken
  cd04e8609a05fb58c1710e99d306a20f  completed
  d9aaabf813d38a25b1646b9c982b59ec  interrupted
  10ef14fdde144550ea9e11d667003bbf  new
  7d80176152e68a0fb758ca61cac15a9a  reserved
  4cd0ee66c65cb32ee20a7b1e2cc47cdc  suspended


"""

    assert captured == expected


def test_three_unrelated_w_a_wout_c(three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --all."""
    orion.core.cli.main(["status", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  be3832db95702cd617eb4215d9eaa7dc  broken
  cd04e8609a05fb58c1710e99d306a20f  completed
  d9aaabf813d38a25b1646b9c982b59ec  interrupted
  10ef14fdde144550ea9e11d667003bbf  new
  7d80176152e68a0fb758ca61cac15a9a  reserved
  4cd0ee66c65cb32ee20a7b1e2cc47cdc  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9a351eb75be219e27327e16dcb2a7b3b  broken
43b457cde7502ccb584f724d4562ed27  completed            0
156d55adbfc36923421cb05da1116a34  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_three_related_w_a_wout_c(three_family_with_trials, capsys):
    """Test three related experiments with --all."""
    orion.core.cli.main(["status", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  be3832db95702cd617eb4215d9eaa7dc  broken
  cd04e8609a05fb58c1710e99d306a20f  completed
  d9aaabf813d38a25b1646b9c982b59ec  interrupted
  10ef14fdde144550ea9e11d667003bbf  new
  7d80176152e68a0fb758ca61cac15a9a  reserved
  4cd0ee66c65cb32ee20a7b1e2cc47cdc  suspended


  test_double_exp_child2-v1
  =========================
  id                                status
  --------------------------------  -----------
  1429f8a8760acdb505b0b38d43f152f4  broken
  1d7095260e0f3d4f41608f86811492c8  completed
  ce4e9e1b07f727e6ac937669cc797d8f  interrupted
  22138467d85ee8b1305b9edab47b30da  new
  74a7c6101b9a454d5da718755ed477df  reserved
  664ef1ab5f7b8714f2bb7d1800b92012  suspended


"""

    assert captured == expected


def test_three_related_branch_w_a_wout_c(three_family_branch_with_trials, capsys):
    """Test three related experiments in a branch with --all."""
    orion.core.cli.main(["status", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  be3832db95702cd617eb4215d9eaa7dc  broken
  cd04e8609a05fb58c1710e99d306a20f  completed
  d9aaabf813d38a25b1646b9c982b59ec  interrupted
  10ef14fdde144550ea9e11d667003bbf  new
  7d80176152e68a0fb758ca61cac15a9a  reserved
  4cd0ee66c65cb32ee20a7b1e2cc47cdc  suspended


    test_double_exp_grand_child-v1
    ==============================
    id                                status
    --------------------------------  -----------
    5d72539d216133efe51cdcb18f91e09a  broken
    d46a4899ea20582496b6d1a8c6e30914  completed
    5e10783a202a53886058d48473399f67  interrupted
    2a3f024a07dae1831f340e40647dc5a9  new
    2bf75d419594525cd21576a2cd124066  reserved
    04b3a8550beac9a595cedf5787f6f984  suspended


"""

    assert captured == expected


def test_two_unrelated_w_c_wout_a(unrelated_with_trials, capsys):
    """Test two unrelated experiments with --collapse."""
    orion.core.cli.main(["status", "--collapse"])

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


def test_two_related_w_c_wout_a(family_with_trials, capsys):
    """Test two related experiments with --collapse."""
    orion.core.cli.main(["status", "--collapse"])

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


def test_three_unrelated_w_c_wout_a(three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --collapse."""
    orion.core.cli.main(["status", "--collapse"])

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


def test_three_related_w_c_wout_a(three_family_with_trials, capsys):
    """Test three related experiments with --collapse."""
    orion.core.cli.main(["status", "--collapse"])

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


def test_three_related_branch_w_c_wout_a(three_family_branch_with_trials, capsys):
    """Test three related experiments with --collapse."""
    orion.core.cli.main(["status", "--collapse"])

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


def test_two_unrelated_w_ac(unrelated_with_trials, capsys):
    """Test two unrelated experiments with --collapse and --all."""
    orion.core.cli.main(["status", "--collapse", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9a351eb75be219e27327e16dcb2a7b3b  broken
43b457cde7502ccb584f724d4562ed27  completed            0
156d55adbfc36923421cb05da1116a34  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_two_related_w_ac(family_with_trials, capsys):
    """Test two related experiments with --collapse and --all."""
    orion.core.cli.main(["status", "--collapse", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
6e15ccc12f754526f73aaf21c4078521  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_three_unrelated_w_ac(three_experiments_with_trials, capsys):
    """Test three unrelated experiments with --collapse and --all."""
    orion.core.cli.main(["status", "--collapse", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
6e15ccc12f754526f73aaf21c4078521  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
9a351eb75be219e27327e16dcb2a7b3b  broken
43b457cde7502ccb584f724d4562ed27  completed            0
156d55adbfc36923421cb05da1116a34  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_three_related_w_ac(three_family_with_trials, capsys):
    """Test three related experiments with --collapse and --all."""
    orion.core.cli.main(["status", "--collapse", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
6e15ccc12f754526f73aaf21c4078521  new
7e6119201c6b7d1b4dcca37768c31782  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_three_related_branch_w_ac(three_family_branch_with_trials, capsys):
    """Test three related experiments in a branch with --collapse and --all."""
    orion.core.cli.main(["status", "--collapse", "--all"])

    captured = capsys.readouterr().out

    expected = """\
test_double_exp-v1
==================
id                                status
--------------------------------  -----------
8032fb2ba95ba375c9f457ae5615b598  broken
156d55adbfc36923421cb05da1116a34  completed
9a351eb75be219e27327e16dcb2a7b3b  interrupted
95a525048fef91ea2cd0ca807ddaa46a  new
6e15ccc12f754526f73aaf21c4078521  new
ad064310b117d0d31edd51507f51b4ca  new
3a6998c7b0d659a272b0fde382d66dee  reserved
de370b6f6db68b3b8060ec76c9715835  suspended


"""

    assert captured == expected


def test_no_experiments_w_name(orionstate, monkeypatch, capsys):
    """Test status when --name <exp> does not exist."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["status", "--name", "test_ghost_exp"])

    captured = capsys.readouterr().out

    assert captured == "No experiment found\n"


def test_experiment_wout_child_w_name(unrelated_with_trials, capsys):
    """Test status with the name argument and no child."""
    orion.core.cli.main(["status", "--name", "test_single_exp"])

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


def test_experiment_w_child_w_name(three_experiments_with_trials, capsys):
    """Test status with the name argument and one child."""
    orion.core.cli.main(["status", "--name", "test_double_exp"])

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


def test_experiment_w_parent_w_name(three_experiments_with_trials, capsys):
    """Test status with the name argument and one parent."""
    orion.core.cli.main(["status", "--name", "test_double_exp_child"])

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


def test_experiment_same_name_wout_exv(three_experiments_same_name, capsys):
    """Test status with three experiments having the same name but different versions."""
    orion.core.cli.main(["status"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v3
==================
empty


"""

    assert captured == expected


def test_experiment_same_name_wout_exv_w_name(three_experiments_same_name, capsys):
    """Test status with three experiments having the same name but different versions."""
    orion.core.cli.main(["status", "--name", "test_single_exp"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v3
==================
empty


"""

    assert captured == expected


def test_experiment_same_name_wout_exv_w_child_w_name(
    three_experiments_family_same_name, capsys
):
    """Test status name with two experiments having the same name and one with a child."""
    orion.core.cli.main(["status", "--name", "test_single_exp"])

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
    three_experiments_family_same_name, capsys
):
    """Test status name collapsed with two experiments having the same name and one with a child."""
    orion.core.cli.main(["status", "--name", "test_single_exp", "--collapse"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v2
==================
empty


"""

    assert captured == expected


def test_experiment_same_name_wout_exv_w_child(
    three_experiments_family_same_name, capsys
):
    """Test status with two experiments having the same name and one with a child."""
    orion.core.cli.main(["status"])

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


def test_experiment_same_name_w_exv(three_experiments_same_name, capsys):
    """Test status with three experiments with the same name and `--expand-verions`."""
    orion.core.cli.main(["status", "--expand-versions"])

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


def test_experiment_same_name_w_exv_w_child(three_experiments_family_same_name, capsys):
    """Test status with two experiments having the same name and one with a child."""
    orion.core.cli.main(["status", "--expand-versions"])

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


def test_experiment_specific_version(three_experiments_same_name, capsys):
    """Test status using `--version`."""
    orion.core.cli.main(["status", "--version", "2"])

    captured = capsys.readouterr().out

    expected = """\
test_single_exp-v2
==================
empty


"""

    assert captured == expected


def test_experiment_cant_use_version(three_experiments_same_name):
    """Test status using `--version`."""
    with pytest.raises(RuntimeError) as ex:
        orion.core.cli.main(["status", "--version", "2", "--collapse"])

    assert "collapse" in str(ex.value)

    with pytest.raises(RuntimeError) as ex:
        orion.core.cli.main(["status", "--version", "2", "--expand-versions"])

    assert "expand-versions" in str(ex.value)
