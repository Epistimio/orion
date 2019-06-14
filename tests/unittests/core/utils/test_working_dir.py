#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.utils.working_dir`."""
import os

from orion.core.utils.working_dir import WorkingDir


def test_create_permanent_dir(tmp_path):
    """Check if a permanent directory is created."""
    with WorkingDir(tmp_path, temp=False, prefix="hi"):
        assert os.path.exists(tmp_path / "hi")

    assert os.path.exists(tmp_path / "hi")


def test_create_temp_dir(tmp_path):
    """Check if a temporary directory is created."""
    with WorkingDir(tmp_path, prefix="hi") as w:
        assert os.path.exists(w)

    assert not os.path.exists(w)
