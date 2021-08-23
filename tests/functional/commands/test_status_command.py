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
  2b08bdbad673e60fef739b7f162d4120  broken
  af45e26f349f9b186e5c05a91d854fb5  completed
  b334ea9e2c86873ddb18b206cf72cc27  interrupted
  16b024079173ca3903eb956c478afa3d  new
  17ce012a15a7398d3e7703d0c13e21c2  reserved
  d4721fe7f50df1fe3ba60424df6dec67  suspended


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
  2b08bdbad673e60fef739b7f162d4120  broken
  af45e26f349f9b186e5c05a91d854fb5  completed
  b334ea9e2c86873ddb18b206cf72cc27  interrupted
  16b024079173ca3903eb956c478afa3d  new
  17ce012a15a7398d3e7703d0c13e21c2  reserved
  d4721fe7f50df1fe3ba60424df6dec67  suspended


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
  2b08bdbad673e60fef739b7f162d4120  broken
  af45e26f349f9b186e5c05a91d854fb5  completed
  b334ea9e2c86873ddb18b206cf72cc27  interrupted
  16b024079173ca3903eb956c478afa3d  new
  17ce012a15a7398d3e7703d0c13e21c2  reserved
  d4721fe7f50df1fe3ba60424df6dec67  suspended


  test_double_exp_child2-v1
  =========================
  id                                status
  --------------------------------  -----------
  f736224f9687f86c493a004696abd95b  broken
  2def838a2eb199820f283e1948e7c37a  completed
  75752e1ba3c9007e42616249087a7fef  interrupted
  5f5e1c8d886ef0b0c0666d6db7bf1723  new
  2623a01bd2483a5e18fac9bc3dfbdee2  reserved
  2eecad70c53bb52c99efad36f2d9502f  suspended


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
  2b08bdbad673e60fef739b7f162d4120  broken
  af45e26f349f9b186e5c05a91d854fb5  completed
  b334ea9e2c86873ddb18b206cf72cc27  interrupted
  16b024079173ca3903eb956c478afa3d  new
  17ce012a15a7398d3e7703d0c13e21c2  reserved
  d4721fe7f50df1fe3ba60424df6dec67  suspended


    test_double_exp_grand_child-v1
    ==============================
    id                                status
    --------------------------------  -----------
    e1c929d9c4d48eca4dcd463690e4096d  broken
    e8eec526a7f7fdea5e4a30d969ec69ae  completed
    e88f1d26158efb393ae1278c6ef115fe  interrupted
    4510dc7d16a692c7415dd2898faced9f  new
    523881db96da0de9dd972ef8f3545f81  reserved
    f967423d15c50ddf88c242f511997ff7  suspended


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
cd16dd40955335aae3bd40371e636b71  new
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
cd16dd40955335aae3bd40371e636b71  new
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
cd16dd40955335aae3bd40371e636b71  new
8d5652cba225224d6702107e97a53cd9  new
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
cd16dd40955335aae3bd40371e636b71  new
17fab2503ac14ae55e207c7cca1b8f1f  new
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
