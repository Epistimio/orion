#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.utils.working_dir`."""
import os
import shutil

import pytest

from orion.core.utils.working_dir import WorkingDir


@pytest.fixture
def path(tmp_path):
    """Return a path as a string."""
    return str(tmp_path) + "/hi_hello"


def test_create_permanent_dir(tmp_path, path):
    """Check if a permanent directory is created."""
    with WorkingDir(tmp_path, temp=False, prefix="hi", suffix="_hello"):
        assert os.path.exists(path)

    assert os.path.exists(path)


def test_temp_dir_when_exists(tmp_path, path):
    """Check if a permanent directory is deleted."""
    os.mkdir(path)

    with WorkingDir(tmp_path, temp=True, prefix="hi", suffix="_hello"):
        assert os.path.exists(path)

    assert os.path.exists(path)

    shutil.rmtree(path)


def test_create_temp_dir(tmp_path):
    """Check if a temporary directory is created."""
    with WorkingDir(tmp_path, prefix="hi", suffix="_hello") as w:
        assert os.path.exists(w)

    assert not os.path.exists(w)
