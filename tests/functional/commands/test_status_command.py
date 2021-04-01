#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the status command."""
import os

import pytest

import orion.core.cli


def test_no_experiments(setup_pickleddb_database, monkeypatch, capsys):
    """Test status with no experiments."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["status"])

    captured = capsys.readouterr().out

    assert captured == "No experiment found\n"


def test_no_version_backward_compatible(one_experiment_no_version, capsys):
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
6c10cf8f7a065c9f0bb0290800ce2926  broken
d8a550d66949a115d3db3925bd93829c  completed            0
ad2f2049de1fe1a600cde6d118588461  interrupted
9cc0d5701543d6eff7a7cd95dd0681b9  new
3de6f76692d763e3ccb1422172cccc1d  reserved
1a9fc30e1bd96b26e086196112f31a69  suspended


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
6c10cf8f7a065c9f0bb0290800ce2926  broken
ad2f2049de1fe1a600cde6d118588461  interrupted
9cc0d5701543d6eff7a7cd95dd0681b9  new
3de6f76692d763e3ccb1422172cccc1d  reserved
1a9fc30e1bd96b26e086196112f31a69  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
6c10cf8f7a065c9f0bb0290800ce2926  broken
d8a550d66949a115d3db3925bd93829c  completed            0
ad2f2049de1fe1a600cde6d118588461  interrupted
9cc0d5701543d6eff7a7cd95dd0681b9  new
3de6f76692d763e3ccb1422172cccc1d  reserved
1a9fc30e1bd96b26e086196112f31a69  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  890b4f07685ed020f5d9e28cac9316e1  broken
  ff81ff46da5ffe6bd623fb38a06df993  completed
  78a3e60699eee1d0b9bc51a049168fce  interrupted
  13cd454155748351790525e3079fb620  new
  d1b7ecbd3621de9195a42c76defb6603  reserved
  33d6208ef03cb236a8f3b567665c357d  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  890b4f07685ed020f5d9e28cac9316e1  broken
  ff81ff46da5ffe6bd623fb38a06df993  completed
  78a3e60699eee1d0b9bc51a049168fce  interrupted
  13cd454155748351790525e3079fb620  new
  d1b7ecbd3621de9195a42c76defb6603  reserved
  33d6208ef03cb236a8f3b567665c357d  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
6c10cf8f7a065c9f0bb0290800ce2926  broken
d8a550d66949a115d3db3925bd93829c  completed            0
ad2f2049de1fe1a600cde6d118588461  interrupted
9cc0d5701543d6eff7a7cd95dd0681b9  new
3de6f76692d763e3ccb1422172cccc1d  reserved
1a9fc30e1bd96b26e086196112f31a69  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  890b4f07685ed020f5d9e28cac9316e1  broken
  ff81ff46da5ffe6bd623fb38a06df993  completed
  78a3e60699eee1d0b9bc51a049168fce  interrupted
  13cd454155748351790525e3079fb620  new
  d1b7ecbd3621de9195a42c76defb6603  reserved
  33d6208ef03cb236a8f3b567665c357d  suspended


  test_double_exp_child2-v1
  =========================
  id                                status
  --------------------------------  -----------
  1c238040d6b6d8423d99a08551fe0998  broken
  2c13424a9212ab92ea592bdaeb1c13e9  completed
  a2680fbda1faa9dfb94946cf25536f44  interrupted
  abbda454d0577ded5b8e784a9d6d5abb  new
  df58aa8fd875f129f7faa84eb15ca453  reserved
  71657e86bad0f2e8b06098a64cb883b6  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


  test_double_exp_child-v1
  ========================
  id                                status
  --------------------------------  -----------
  890b4f07685ed020f5d9e28cac9316e1  broken
  ff81ff46da5ffe6bd623fb38a06df993  completed
  78a3e60699eee1d0b9bc51a049168fce  interrupted
  13cd454155748351790525e3079fb620  new
  d1b7ecbd3621de9195a42c76defb6603  reserved
  33d6208ef03cb236a8f3b567665c357d  suspended


    test_double_exp_grand_child-v1
    ==============================
    id                                status
    --------------------------------  -----------
    e374d8f802aed52c07763545f46228a7  broken
    f9ee14ff9ef0b95ed7a24860731c85a9  completed
    3f7dff101490727d5fa0efeb36ca6366  interrupted
    40838d46dbf7778a3cb51b7a09118391  new
    b7860a18b2700cce4e8009cde543975c  reserved
    cd406126bc350ad82ac77c75174cc8a2  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
6c10cf8f7a065c9f0bb0290800ce2926  broken
d8a550d66949a115d3db3925bd93829c  completed            0
ad2f2049de1fe1a600cde6d118588461  interrupted
9cc0d5701543d6eff7a7cd95dd0681b9  new
3de6f76692d763e3ccb1422172cccc1d  reserved
1a9fc30e1bd96b26e086196112f31a69  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
4c409da13bdc93c54f6997797c296356  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
4c409da13bdc93c54f6997797c296356  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


test_single_exp-v1
==================
id                                status         min obj
--------------------------------  -----------  ---------
6c10cf8f7a065c9f0bb0290800ce2926  broken
d8a550d66949a115d3db3925bd93829c  completed            0
ad2f2049de1fe1a600cde6d118588461  interrupted
9cc0d5701543d6eff7a7cd95dd0681b9  new
3de6f76692d763e3ccb1422172cccc1d  reserved
1a9fc30e1bd96b26e086196112f31a69  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
4c409da13bdc93c54f6997797c296356  new
b97518f91e006cd4a2805657c596b11c  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


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
259021a55c2a5af7074a41ed6639b2f3  broken
cbb766d729294f77f0ca86ff2bf72707  completed
ca6576848f17201852225d816fb71fcc  interrupted
28097ba31dbdffc0aa265c6bc5c98b0f  new
4c409da13bdc93c54f6997797c296356  new
5183ee9c28601cc78c0a148a386df9f9  new
adbe6c400cd1e667696e28fbecd000a0  reserved
5679af6c6bb54aa8042043008ab2bc1f  suspended


"""

    assert captured == expected


def test_no_experiments_w_name(setup_pickleddb_database, monkeypatch, capsys):
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
