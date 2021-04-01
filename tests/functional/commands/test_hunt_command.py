#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the hunt command."""
import pytest

import orion.core.cli


def test_hunt_no_prior(one_experiment, capsys):
    """Test at least one prior is specified"""
    orion.core.cli.main(["hunt", "-n", "test", "./black_box.py"])

    captured = capsys.readouterr().err

    assert "No prior found" in captured
    assert "Traceback" not in captured


def test_no_args(capsys):
    """Test that help is printed when no args are given."""
    with pytest.raises(SystemExit):
        orion.core.cli.main(["hunt"])

    captured = capsys.readouterr().out

    assert "usage:" in captured
    assert "Traceback" not in captured


def test_no_name(capsys):
    """Try to run the command without providing an experiment name"""
    returncode = orion.core.cli.main(["hunt", "--exp-max-trials", "10"])
    assert returncode == 1

    captured = capsys.readouterr().err

    assert captured == "Error: No name provided for the experiment.\n"
