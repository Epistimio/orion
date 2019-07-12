#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the list command."""
import os

import pytest

import orion.core.cli
from orion.core.io.database import Database


@pytest.fixture
def no_experiment(database):
    """Create and save a singleton for an empty database instance."""
    database.experiments.drop()
    database.lying_trials.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()

    try:
        db = Database(of_type='MongoDB', name='orion_test',
                      username='user', password='pass')
    except ValueError:
        db = Database()

    return db


@pytest.fixture
def one_experiment(monkeypatch, create_db_instance):
    """Create a single experiment."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_list_single',
                         './black_box.py', '--x~uniform(0,1)'])


@pytest.fixture
def two_experiments(monkeypatch, no_experiment):
    """Create an experiment and its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_list_double',
                         './black_box.py', '--x~uniform(0,1)'])
    orion.core.cli.main(['init_only', '-n', 'test_list_double',
                         '--branch', 'test_list_double_child', './black_box.py',
                         '--x~uniform(0,1)', '--y~+uniform(0,1)'])


@pytest.fixture
def three_experiments(monkeypatch, two_experiments):
    """Create a single experiment and an experiment and its child."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(['init_only', '-n', 'test_list_single',
                         './black_box.py', '--x~uniform(0,1)'])


def test_no_exp(no_experiment, monkeypatch, capsys):
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
