#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the list command."""
import os

import pytest

import orion.core.cli


@pytest.fixture
def no_experiment(database):
    """Make sure there is no experiment."""
    database.experiments.drop()


@pytest.fixture
def one_experiment(monkeypatch, no_experiment):
    """Create a single experiment."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_list_single', '-c', './orion_config_random.yaml',
                         './black_box.py', '--x~uniform(0,1)'])


@pytest.fixture
def two_experiments(monkeypatch, no_experiment):
    """Create an experiment and its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_list_double', '-c', './orion_config_random.yaml',
                         './black_box.py', '--x~uniform(0,1)'])
    orion.core.cli.main(['init_only', '-n', 'test_list_double', '-c', './orion_config_random.yaml',
                         '--branch', 'test_list_double_child', './black_box.py',
                         '--x~uniform(0,1)', '--y~+uniform(0,1)'])


@pytest.fixture
def three_experiments(monkeypatch, two_experiments):
    """Create a single experiment and an experiment and its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_list_single', '-c', './orion_config_random.yaml',
                         './black_box.py', '--x~uniform(0,1)'])


def test_no_exp(no_experiment, monkeypatch, capsys):
    """Test that nothing is printed when there are no experiments."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['list', '--config', './orion_config_random.yaml'])

    captured = capsys.readouterr().out

    assert captured == ""


def test_single_exp(capsys, one_experiment):
    """Test that the name of the experiment is printed when there is one experiment."""
    orion.core.cli.main(['list', '--config', './orion_config_random.yaml'])

    captured = capsys.readouterr().out

    assert captured == " test_list_single\n"


def test_two_exp(capsys, two_experiments):
    """Test that nothing is printed when there are no experiments."""
    orion.core.cli.main(['list', '--config', './orion_config_random.yaml'])

    captured = capsys.readouterr().out

    assert captured == " test_list_double┐\n                 └test_list_double_child\n"


def test_three_exp(capsys, three_experiments):
    """Test that nothing is printed when there are no experiments."""
    orion.core.cli.main(['list', '--config', './orion_config_random.yaml'])

    captured = capsys.readouterr().out

    assert captured == " test_list_double┐\n                 └test_list_double_child\n \
test_list_single\n"
