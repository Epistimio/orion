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


def test_user_script_crash(capfd):
    """Checks that the Traceback of a crashing user script is output to stderr"""

    orion.core.cli.main(
        [
            "--debug",
            "hunt",
            "-n",
            "test",
            "--exp-max-trials",
            "1",
            "./black_box_fail.py",
            "-c",
            "-x~uniform(-5,5)",
        ]
    )

    out, err = capfd.readouterr()

    assert "Traceback" in err
    assert "# black_box_fail.py : this line should crash" in err
