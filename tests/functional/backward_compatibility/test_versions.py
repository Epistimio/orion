#!/usr/bin/env python
"""Perform functional tests to verify backward compatibility."""
import os
import shutil
import subprocess

import pytest
from pymongo import MongoClient

import orion.core.io.experiment_builder as experiment_builder
from orion.client import create_experiment
from orion.core.io.database import database_factory
from orion.storage.base import storage_factory

DIRNAME = os.path.dirname(os.path.abspath(__file__))

PYTHON_SCRIPT_PATH = os.path.join(DIRNAME, "python_api.py")
SCRIPT_PATH = os.path.join(DIRNAME, "black_box.py")
CONFIG_FILE = os.path.join(DIRNAME, "random.yaml")

# Ignore pre-0.1.6 because was on orion.core and pypi project was deleted.
with open(os.path.join(DIRNAME, "versions.txt")) as f:
    VERSIONS = [version.strip() for version in f.read().split("\n") if version.strip()]


def function(x):
    """Evaluate partial information of a quadratic."""
    z = x - 34.56789
    return [dict(name="example_objective", type="objective", value=4 * z**2 + 23.4)]


def get_branch_argument(version):
    """Get argument to branch.

    Before v0.1.8 it was --branch. From v0.1.8 and forward it is now --branch-to.
    """
    return "--branch" if version in ["0.1.6", "0.1.7"] else "--branch-to"


def get_evc_argument(version):
    """Get argument to enable EVC

    Before v0.1.16 EVC was enabled by default. Starting from v0.1.16 it must be enabled with
    --enable-evc.
    """
    major, minor, patch = list(map(int, version.split(".")[:3]))
    return "--enable-evc" if (major > 0 or minor > 1 or patch > 15) else ""


def has_python_api(version):
    """Whether the python api exist in given version"""
    return version not in ["0.1.6", "0.1.7"]


def clean_mongodb():
    """Clean collections."""
    client = MongoClient(username="user", password="pass", authSource="orion_test")
    database = client.orion_test
    database.experiments.drop()
    database.lying_trials.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()
    client.close()


def set_env():
    """Set the env vars for the db.

    Notes
    -----
    This function expects the env var ORION_DB_TYPE to be set.

    """
    db_type = os.environ["ORION_DB_TYPE"]
    if db_type == "pickleddb":
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save.pkl")
        os.environ["ORION_DB_ADDRESS"] = file_path
        if os.path.exists(file_path):
            os.remove(file_path)
    elif db_type == "mongodb":
        os.environ["ORION_DB_ADDRESS"] = "mongodb://user:pass@localhost"
        os.environ["ORION_DB_NAME"] = "orion_test"
        clean_mongodb()


def execute(command):
    """Execute a command and return the decoded stdout."""
    out = subprocess.check_output(command.split())
    return out.decode("utf-8")


def setup_virtualenv(version):
    """Create a virtualenv and install Oríon in it for the given version"""
    virtualenv_dir = get_virtualenv_dir(version)
    if os.path.exists(virtualenv_dir):
        shutil.rmtree(virtualenv_dir)
    execute(f"virtualenv {virtualenv_dir}")
    command = "{}/bin/pip install --ignore-installed orion=={}".format(
        virtualenv_dir, version
    )
    execute(command)


def get_version(orion_script):
    """Get Oríon version for given Oríon executer's path."""
    command = f"{orion_script} --version"
    stdout = subprocess.check_output(command, shell=True)

    return stdout.decode("utf-8").strip("\n")


def get_virtualenv_dir(version):
    """Get virtualenv directory for given version."""
    return f"version-{version}"


def fill_from_cmdline_api(orion_script, version):
    """Add experiments and trials using the commandline API"""
    # TODO(v0.3.0): Adapt when removing init_only after deprecation phase
    print(
        execute(
            " ".join(
                [
                    orion_script,
                    "-vv",
                    "init_only",
                    "--name",
                    "init-cmdline",
                    "--config",
                    CONFIG_FILE,
                    SCRIPT_PATH,
                    "-x~uniform(-50,50)",
                ]
            )
        )
    )

    # TODO(v0.3.0): Adapt when removing init_only after deprecation phase
    print(
        execute(
            " ".join(
                [
                    orion_script,
                    "-vv",
                    "init_only",
                    "--name",
                    "init-cmdline",
                    get_evc_argument(version),
                    get_branch_argument(version),
                    "init-cmdline-branch-old",
                    "--config",
                    CONFIG_FILE,
                ]
            )
        )
    )

    print(
        execute(
            " ".join(
                [
                    orion_script,
                    "-vv",
                    "hunt",
                    "--name",
                    "hunt-cmdline",
                    "--config",
                    CONFIG_FILE,
                    SCRIPT_PATH,
                    "-x~uniform(-50,50)",
                ]
            )
        )
    )

    print(
        execute(
            " ".join(
                [
                    orion_script,
                    "-vv",
                    "hunt",
                    "--name",
                    "hunt-cmdline",
                    get_evc_argument(version),
                    get_branch_argument(version),
                    "hunt-cmdline-branch-old",
                    "--config",
                    CONFIG_FILE,
                ]
            )
        )
    )


def fill_from_python_api(python_script, version):
    """Add experiments and trials using the python API"""
    print(execute(" ".join([python_script, PYTHON_SCRIPT_PATH, version])))


@pytest.fixture(scope="class", params=VERSIONS)
def fill_db(request):
    """Add experiments and trials in DB for given version of Oríon."""
    set_env()

    version = request.param

    setup_virtualenv(version)

    orion_script = os.path.join(get_virtualenv_dir(version), "bin", "orion")
    python_script = os.path.join(get_virtualenv_dir(version), "bin", "python")

    orion_version = get_version(orion_script)
    assert orion_version == f"orion {version}"

    fill_from_cmdline_api(orion_script, version)
    if has_python_api(version):
        fill_from_python_api(python_script, version)

    orion_version = get_version("orion")
    assert orion_version != f"orion {version}"

    print(execute("orion -vv db upgrade -f"))

    return version


def null_db_instances():
    """Nullify singleton instance so that we can assure independent instantiation tests."""
    storage_factory.instance = None
    database_factory.instance = None


def build_storage():
    """Build storage from scratch"""
    null_db_instances()
    return experiment_builder.setup_storage()


@pytest.mark.usefixtures("fill_db")
class TestBackwardCompatibility:
    """Tests for backward compatibility between Oríon versions."""

    def test_db_upgrade(self):
        """Verify db upgrade was successful"""
        storage = build_storage()

        index_info = storage._db.index_information("experiments")
        assert {key for key, is_unique in index_info.items() if is_unique} == {
            "_id_",
            "name_1_version_1",
        }

        experiments = storage.fetch_experiments({})
        assert "version" in experiments[0]

        trials = storage.fetch_trials(uid=experiments[0]["_id"])
        for trial in trials:
            assert trial.id_override is not None
            trial_2 = storage.get_trial(
                uid=trial.id, experiment_uid=experiments[0]["_id"]
            )
            assert trial == trial_2

    def test_db_test(self):
        """Verify db test command"""
        out = execute("orion db test")
        assert "Failure" not in out

    def test_list(self, fill_db):
        """Verify list command"""
        out = execute("orion list")
        assert "init-cmdline-v1" in out
        assert "init-cmdline-branch-old-v1" in out
        assert "hunt-cmdline-v1" in out
        assert "hunt-cmdline-branch-old-v1" in out

        version = fill_db
        if has_python_api(version):
            assert "hunt-python-v1" in out

    def test_status(self, fill_db):
        """Verify status command"""
        out = execute("orion status")
        assert "init-cmdline-v1" in out
        assert "init-cmdline-branch-old-v1" in out
        assert "hunt-cmdline-v1" in out
        assert "hunt-cmdline-branch-old-v1" in out

        version = fill_db
        if has_python_api(version):
            assert "hunt-python-v1" in out

    def test_info_cmdline_api(self):
        """Verify info command from commandline api"""
        out = execute("orion info --name hunt-cmdline")
        assert "name: hunt-cmdline" in out

    def test_info_python_api(self, fill_db):
        """Verify info command from python api"""
        version = fill_db
        if not has_python_api(version):
            pytest.skip(f"Python API not supported by {version}")

        out = execute("orion info --name hunt-python")
        assert "name: hunt-python" in out

    # TODO(v0.3.0): Adapt when removing init_only after deprecation phase
    def test_init_only(self):
        """Verify init_only command"""
        print(
            execute(
                " ".join(
                    [
                        "orion",
                        "init_only",
                        "--name",
                        "init-cmdline",
                        "--branch-to",
                        "init-cmdline-branch",
                    ]
                )
            )
        )

    def test_hunt_cmdline_api(self):
        """Verify hunt command from cmdline api parent"""
        print(
            execute(
                " ".join(
                    [
                        "orion",
                        "hunt",
                        "--name",
                        "hunt-cmdline",
                        "--branch-to",
                        "hunt-cmdline-branch",
                    ]
                )
            )
        )

    def test_hunt_python_api(self, fill_db):
        """Verify hunt command from python api parent"""
        version = fill_db
        if not has_python_api(version):
            pytest.skip(f"Python API not supported by {version}")

        exp = create_experiment(
            "hunt-python", branching={"branch-to": "hunt-python-branch"}
        )
        exp.workon(function, max_trials=10)
