#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""perform a functional test of the list command."""
import os

import pytest

import orion.core.cli


@pytest.mark.usefixtures('clean_db')
def test_no_exp(monkeypatch, capsys):
    """Test that nothing is printed when there are no experiments."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == ""


def test_single_exp(capsys, one_experiment):
    """Test that the name of the experiment is printed when there is one experiment."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_list_single\n"


def test_two_exp(capsys, two_experiments):
    """Test that experiment and child are printed."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_list_double┐\n                 └test_list_double_child\n"


def test_three_exp(capsys, three_experiments):
    """Test that experiment, child  and grand-child are printed."""
    orion.core.cli.main(['list'])

    captured = capsys.readouterr().out

    assert captured == " test_list_double┐\n                 └test_list_double_child\n \
test_list_single\n"


def test_no_exp_name(three_experiments, monkeypatch, capsys):
    """Test that nothing is printed when there are no experiments with a given name."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'I don\'t exist'])

    captured = capsys.readouterr().out

    assert captured == ""


def test_exp_name(three_experiments, monkeypatch, capsys):
    """Test that only the specified experiment is printed."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_list_single'])

    captured = capsys.readouterr().out

    assert captured == " test_list_single\n"


def test_exp_name_with_child(three_experiments, monkeypatch, capsys):
    """Test that only the specified experiment is printed, and with its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_list_double'])

    captured = capsys.readouterr().out

    assert captured == " test_list_double┐\n                 └test_list_double_child\n"


def test_exp_name_child(three_experiments, monkeypatch, capsys):
    """Test that only the specified child experiment is printed."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--name', 'test_list_double_child'])

    captured = capsys.readouterr().out

    assert captured == " test_list_double_child\n"
