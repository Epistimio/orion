#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.utils.working_dir`."""
import os
import shutil
from pathlib import Path

from orion.core.utils.working_dir import SetupWorkingDir


class ExperimentStub:
    def __init__(self, working_dir=None):
        self.name = "exp-name"
        self.version = 1
        self.working_dir = working_dir


def test_exp_with_new_working_dir(tmp_path):
    """Check if a permanent directory is created."""
    tmp_path = os.path.join(tmp_path, "orion")

    experiment = ExperimentStub(tmp_path)

    assert not os.path.exists(tmp_path)

    with SetupWorkingDir(experiment):
        assert os.path.exists(tmp_path)

    assert experiment.working_dir == tmp_path
    assert os.path.exists(tmp_path)

    shutil.rmtree(tmp_path)


def test_exp_with_existing_working_dir(tmp_path):
    """Check if an existing permanent directory is not overwritten."""
    tmp_path = os.path.join(tmp_path, "orion")

    experiment = ExperimentStub(tmp_path)

    os.makedirs(tmp_path)

    assert os.path.exists(tmp_path)

    file_path = os.path.join(tmp_path, "some_file")
    Path(file_path).touch()

    assert os.path.exists(file_path)

    with SetupWorkingDir(experiment):
        assert os.path.exists(tmp_path)

    assert experiment.working_dir == tmp_path
    assert os.path.exists(tmp_path)
    assert os.path.exists(file_path)

    shutil.rmtree(tmp_path)


def test_exp_with_no_working_dir():
    """Check if a permanent directory is deleted."""
    experiment = ExperimentStub(None)

    with SetupWorkingDir(experiment):
        assert experiment.working_dir is not None
        assert os.path.exists(experiment.working_dir)
        tmp_path = experiment.working_dir

    assert experiment.working_dir is None
    assert not os.path.exists(tmp_path)
