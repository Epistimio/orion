#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the init_only command."""
import orion.core.cli


def test_no_args(capsys):
    """Try to run the command without any arguments"""
    returncode = orion.core.cli.main(["init_only"])
    assert returncode == 1

    captured = capsys.readouterr().err

    assert captured == 'Error: No name provided for the experiment.\n'
