#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the info command."""
import orion.core.cli


def test_info_no_hit(clean_db, one_experiment, capsys):
    """Test info if no experiment with given name."""
    orion.core.cli.main(['info','--name' , 'i do not exist'])

    captured = capsys.readouterr().out

    assert captured == 'Error: No commandline configuration found for new experiment.\n'


def test_info_hit(clean_db, one_experiment, capsys):
    """Test info if existing experiment."""
    orion.core.cli.main(['info', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    assert '--x~uniform(0,1)' in captured


def test_info_broken(clean_db, broken_refers, capsys):
    """Test info if experiment.refers is missing."""
    orion.core.cli.main(['info', '--name', 'test_single_exp'])

    captured = capsys.readouterr().out

    assert '--x~uniform(0,1)' in captured
