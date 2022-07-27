#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.evc.cli`."""
import argparse

import pytest

from orion.core.cli.evc import get_branching_args_group
from orion.core.evc.conflicts import Resolution


def test_defined_parser():
    """Test that all expected branching arguments are present"""
    parser = argparse.ArgumentParser()
    get_branching_args_group(parser)

    options = parser.parse_args([])
    assert options.manual_resolution is None
    assert options.algorithm_change is None
    assert options.branch_to is None
    assert options.branch_from is None
    assert options.cli_change_type is None
    assert options.code_change_type is None
    assert options.config_change_type is None


def test_undefined_parser():
    """Test that creation of new resolution class make parser creation crash"""
    # It works
    get_branching_args_group(argparse.ArgumentParser())

    class DummyResolution(Resolution):
        ARGUMENT = "--dummy"

    # It doesn't
    with pytest.raises(AssertionError) as exc:
        get_branching_args_group(argparse.ArgumentParser())
    assert "A resolution with metavar 'dummy'" in str(exc.value)
