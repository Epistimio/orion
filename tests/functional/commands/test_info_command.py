#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the info command."""
import orion.core.cli
import orion.core.io.resolve_config


def test_info_no_hit(one_experiment, capsys):
    """Test info if no experiment with given name."""
    returncode = orion.core.cli.main(["info", "--name", "i do not exist"])

    assert returncode == 1

    captured = capsys.readouterr().err

    assert captured.startswith("Error: No experiment with given name 'i do not exist'")


def test_info_hit(one_experiment, capsys):
    """Test info if existing experiment."""
    orion.core.cli.main(["info", "--name", "test_single_exp"])

    captured = capsys.readouterr().out

    assert "--x~uniform(0,1)" in captured


def test_info_broken(broken_refers, capsys):
    """Test info if experiment.refers is missing."""
    orion.core.cli.main(["info", "--name", "test_single_exp"])

    captured = capsys.readouterr().out

    assert "--x~uniform(0,1)" in captured


def test_info_no_branching(one_experiment_changed_vcs, capsys):
    """Test info if config file is different

    Version should not increase!
    """
    orion.core.cli.main(["info", "--name", "test_single_exp"])

    captured = capsys.readouterr().out

    assert "\nversion: 1\n" in captured


def test_info_python_api(with_experiment_using_python_api, capsys):
    """Test info if config built using python api"""
    orion.core.cli.main(["info", "--name", "from-python-api"])

    captured = capsys.readouterr().out

    assert "from-python-api" in captured
    assert "Commandline" not in captured


def test_missing_conf_file(with_experiment_missing_conf_file, capsys):
    """Test info can handle experiments when the user script config file is missing"""
    orion.core.cli.main(["info", "--name", "test_single_exp"])

    captured = capsys.readouterr().out

    assert "--x~uniform(0,1)" in captured


def test_info_cmdline_api(one_experiment, capsys):
    """Test info if config built using cmdline api"""
    orion.core.cli.main(["info", "--name", "test_single_exp"])

    captured = capsys.readouterr().out

    assert "test_single_exp" in captured
    assert "Commandline" in captured


def test_no_args(capsys):
    """Try to run the command without any arguments"""
    returncode = orion.core.cli.main(["info"])
    assert returncode == 1

    captured = capsys.readouterr().err

    assert captured == "Error: No name provided for the experiment.\n"
