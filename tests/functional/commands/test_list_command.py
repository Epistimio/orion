#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the list command."""
import os

import orion.core.cli


def test_no_exp(monkeypatch, clean_db, capsys):
    """Test that nothing is printed when there are no experiments."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == "No experiment found\n"


def test_single_exp(clean_db, one_experiment, capsys):
    """Test that the name of the experiment is printed when there is one experiment."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp-v1\n"


def test_no_version_backward_compatible(clean_db, one_experiment_no_version, capsys):
    """Test status with no experiments."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp-no-version-v1\n"


def test_broken_refers(clean_db, broken_refers, capsys):
    """Test that experiment without refers dict can be handled properly."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp-v1\n"


def test_python_api(clean_db, with_experiment_using_python_api, capsys):
    """Test list if containing exps from cmdline api and python api"""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp-v1\n from-python-api-v1\n"


def test_two_exp(capsys, clean_db, two_experiments):
    """Test that experiment and child are printed."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == """\
 test_double_exp-v1┐
                   └test_double_exp_child-v1
"""


def test_three_exp(capsys, clean_db, three_experiments):
    """Test that experiment, child  and grand-child are printed."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == """\
 test_double_exp-v1┐
                   └test_double_exp_child-v1
 test_single_exp-v1
"""


def test_no_exp_name(clean_db, three_experiments, monkeypatch, capsys):
    """Test that nothing is printed when there are no experiments with a given name."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'I don\'t exist'])

    captured = capsys.readouterr().out

    assert captured == "No experiment found\n"


def test_exp_name(clean_db, three_experiments, monkeypatch, capsys):
    """Test that only the specified experiment is printed."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp-v1\n"


def test_exp_name_with_child(clean_db, three_experiments, monkeypatch, capsys):
    """Test that only the specified experiment is printed, and with its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_double_exp'])

    captured = capsys.readouterr().out

    assert captured == """\
 test_double_exp-v1┐
                   └test_double_exp_child-v1
"""


def test_exp_name_child(clean_db, three_experiments, monkeypatch, capsys):
    """Test that only the specified child experiment is printed."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_double_exp_child'])

    captured = capsys.readouterr().out

    assert captured == " test_double_exp_child-v1\n"


def test_exp_same_name(clean_db, two_experiments_same_name, monkeypatch, capsys):
    """Test that two experiments with the same name and different versions are correctly printed."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == """\
 test_single_exp-v1┐
                   └test_single_exp-v2
"""


def test_exp_family_same_name(clean_db, three_experiments_family_same_name, monkeypatch, capsys):
    """Test that two experiments with the same name and different versions are correctly printed
    even when one of them has a child.
    """
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == """\
                   ┌test_single_exp-v2
 test_single_exp-v1┤
                   └test_single_exp_child-v1
"""


def test_exp_family_branch_same_name(clean_db, three_experiments_branch_same_name,
                                     monkeypatch, capsys):
    """Test that two experiments with the same name and different versions are correctly printed
    even when last one has a child.
    """
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == """\
 test_single_exp-v1┐
                   └test_single_exp-v2┐
                                      └test_single_exp_child-v1
"""
