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

    assert captured == ""


def test_single_exp(clean_db, one_experiment, capsys):
    """Test that the name of the experiment is printed when there is one experiment."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp\n"


def test_broken_refers(clean_db, broken_refers, capsys):
    """Test that experiment without refers dict can be handled properly."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp\n"


def test_two_exp(capsys, clean_db, two_experiments):
    """Test that experiment and child are printed."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_double_exp┐\n                └test_double_exp_child\n"


def test_three_exp(capsys, clean_db, three_experiments):
    """Test that experiment, child  and grand-child are printed."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_double_exp┐\n                └test_double_exp_child\n \
test_single_exp\n"


def test_no_exp_name(clean_db, three_experiments, monkeypatch, capsys):
    """Test that nothing is printed when there are no experiments with a given name."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'I don\'t exist'])

    captured = capsys.readouterr().out

    assert captured == ""


def test_exp_name(clean_db, three_experiments, monkeypatch, capsys):
    """Test that only the specified experiment is printed."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    assert captured == " test_single_exp\n"


def test_exp_name_with_child(clean_db, three_experiments, monkeypatch, capsys):
    """Test that only the specified experiment is printed, and with its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_double_exp'])

    captured = capsys.readouterr().out

    assert captured == " test_double_exp┐\n                └test_double_exp_child\n"


def test_exp_name_child(clean_db, three_experiments, monkeypatch, capsys):
    """Test that only the specified child experiment is printed."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_double_exp_child'])

    captured = capsys.readouterr().out

    assert captured == " test_double_exp_child\n"
