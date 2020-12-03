#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test of the db commands."""
import pytest

import orion.core.cli


def test_no_args(capsys):
    """Test that help is printed when no args are given."""
    with pytest.raises(SystemExit):
        orion.core.cli.main(["db"])

    captured = capsys.readouterr().out

    assert "usage:" in captured
    assert "Traceback" not in captured
