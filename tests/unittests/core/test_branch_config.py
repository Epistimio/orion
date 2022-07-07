#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.io.experiment_branch_builder`."""

import copy
import os

import pytest
import yaml

import orion.core
import orion.core.utils.backward as backward
from orion.core import evc
from orion.core.evc.conflicts import (
    AlgorithmConflict,
    ChangedDimensionConflict,
    CodeConflict,
    CommandLineConflict,
    ExperimentNameConflict,
    MissingDimensionConflict,
    NewDimensionConflict,
    OrionVersionConflict,
    ScriptConfigConflict,
    detect_conflicts,
)
from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder


def filter_true(c):
    """Filter solved conflicts"""
    return c.is_resolved is True


def filter_false(c):
    """Filter unsolved conflicts"""
    return not filter_true(c)


@pytest.fixture
def user_config():
    """Generate data dict for user's script's configuration file"""
    data = {
        "a": "orion~uniform(-10,10)",
        "some_other": "test",
        "b": "orion~normal(0,1)",
        "argument": "value",
    }
    return data


@pytest.fixture
def parent_config(user_config):
    """Create a configuration that will not hit the database."""
    user_script = "tests/functional/demo/black_box.py"
    config = dict(
        _id="test",
        name="test",
        algorithms="random",
        version=1,
        metadata={
            "VCS": {
                "type": "git",
                "is_dirty": False,
                "HEAD_sha": "test",
                "active_branch": None,
                "diff_sha": "diff",
            },
            "user_script": user_script,
            "user_args": [
                user_script,
                "--nameless=option",
                "-x~uniform(0,1)",
                "-y~normal(0,1)",
                "-z~uniform(0,10)",
                "--manual-resolution",
            ],
            "user": "some_user_name",
            "orion_version": "XYZ",
        },
        refers={},
    )

    config_file_path = "./parent_config.yaml"

    with open(config_file_path, "w") as f:
        yaml.dump(user_config, f)

    config["metadata"]["user_args"].append("--config=%s" % config_file_path)

    backward.populate_space(config)

    yield config
    os.remove(config_file_path)


@pytest.fixture
def child_config(parent_config, storage):
    """Create a child branching from the test experiment"""
    config = copy.deepcopy(parent_config)
    config["_id"] = "test2"
    config["version"] = 2
    config["refers"]["parent_id"] = parent_config["_id"]
    return config


@pytest.fixture
def missing_config(child_config):
    """Create a child config with a missing dimension"""
    del child_config["metadata"]["user_args"][2]  # del -x
    del child_config["metadata"]["user_args"][2]  # del -y
    backward.populate_space(child_config)
    return child_config


@pytest.fixture
def new_config_with_w(child_config):
    """Create a child config with a new dimension"""
    child_config["metadata"]["user_args"].append("-w_d~normal(0,1)")
    backward.populate_space(child_config)
    return child_config


@pytest.fixture
def changed_config(child_config):
    """Create a child config with a changed dimension"""
    second_element = child_config["metadata"]["user_args"][3]
    second_element = second_element.replace("normal", "uniform")
    child_config["metadata"]["user_args"][3] = second_element
    backward.populate_space(child_config)
    return child_config


@pytest.fixture
def changed_algo_config(child_config):
    """Create a child config with a new algo"""
    child_config["algorithms"] = "stupid-grid"
    return child_config


@pytest.fixture
def changed_orion_version_config(child_config):
    """Create a child config with new orion version"""
    child_config["metadata"]["orion_version"] = "UVW"
    return child_config


@pytest.fixture
def changed_code_config(child_config):
    """Create a child config with a changed dimension"""
    child_config["metadata"]["VCS"]["HEAD_sha"] = "new_test"
    return child_config


@pytest.fixture
def same_userconfig_config(user_config, child_config):
    """Create a child config with a changed dimension"""
    config_file_path = "./same_config.yaml"
    with open(config_file_path, "w") as f:
        yaml.dump(user_config, f)
    child_config["metadata"]["user_args"][-1] = "--config=%s" % config_file_path
    backward.populate_space(child_config)
    yield child_config
    os.remove(config_file_path)


@pytest.fixture
def changed_userconfig_config(user_config, child_config):
    """Create a child config with a changed dimension"""
    config_file_path = "./changed_config.yaml"
    user_config["b"] = "orion~uniform(-20, 0, precision=None)"
    user_config["some_other"] = "hello"
    with open(config_file_path, "w") as f:
        yaml.dump(user_config, f)
    child_config["metadata"]["user_args"][-1] = "--config=%s" % config_file_path
    backward.populate_space(child_config)
    yield child_config
    os.remove(config_file_path)


@pytest.fixture
def changed_cli_config(child_config):
    """Create a child config with a changed dimension"""
    child_config["metadata"]["user_args"] += ["-u=0", "--another=test", "positional"]
    backward.populate_space(child_config)
    return child_config


@pytest.fixture
def list_arg_with_equals_cli_config(child_config):
    """Create a child config with an argument of the
    form --args=1 --args=2 --args=3
    """
    child_config["metadata"]["user_args"] += ["--args=1", "--args=2", "--args=3"]
    backward.populate_space(child_config)
    return child_config


@pytest.fixture
def cl_config():
    """Create a child config with markers for commandline solving"""
    user_script = "tests/functional/demo/black_box.py"
    config = dict(
        name="test",
        branch="test2",
        algorithms="random",
        metadata={
            "hash_commit": "old",
            "user_script": user_script,
            "user_args": [
                user_script,
                "--nameless=option",
                "-x~>w_d",
                "-w_d~+normal(0,1)",
                "-y~+uniform(0,1)",
                "-z~-",
                "--omega~+normal(0,1)",
            ],
            "user": "some_user_name",
            "orion_version": "XYZ",
        },
    )
    backward.populate_space(config)
    return config


@pytest.fixture
def conflicts(
    new_dimension_conflict,
    changed_dimension_conflict,
    missing_dimension_conflict,
    algorithm_conflict,
    orion_version_conflict,
    code_conflict,
    experiment_name_conflict,
    config_conflict,
    cli_conflict,
):
    """Create a container for conflicts with one of each types for testing purposes"""
    conflicts = evc.conflicts.Conflicts()
    conflicts.register(new_dimension_conflict)
    conflicts.register(changed_dimension_conflict)
    conflicts.register(missing_dimension_conflict)
    conflicts.register(algorithm_conflict)
    conflicts.register(orion_version_conflict)
    conflicts.register(code_conflict)
    conflicts.register(experiment_name_conflict)
    conflicts.register(config_conflict)
    conflicts.register(cli_conflict)
    return conflicts


class TestConflictDetection:
    """Test detection of conflicts between two configurations"""

    def test_no_conflicts(self, parent_config, child_config):
        """Test the case where the child is the same as the parent"""
        conflicts = detect_conflicts(parent_config, child_config)

        assert len(conflicts.get()) == 1
        assert isinstance(conflicts.get()[0], ExperimentNameConflict)

    def test_missing_dim_conflict(self, parent_config, missing_config):
        """Test if missing dimension is currently detected"""
        conflicts = detect_conflicts(parent_config, missing_config)

        assert len(conflicts.get()) == 3
        conflict = conflicts.get()[1]

        assert conflict.is_resolved is False
        assert conflict.dimension.name == "/x"
        assert isinstance(conflict, MissingDimensionConflict)

    def test_new_dim_conflict(self, parent_config, new_config_with_w):
        """Test if new dimension is currently detected"""
        conflicts = detect_conflicts(parent_config, new_config_with_w)

        assert len(conflicts.get()) == 2
        conflict = conflicts.get()[1]

        assert conflict.is_resolved is False
        assert conflict.dimension.name == "/w_d"
        assert isinstance(conflict, NewDimensionConflict)

    def test_changed_dim_conflict(self, parent_config, changed_config):
        """Test if changed dimension is currently detected"""
        conflicts = detect_conflicts(parent_config, changed_config)

        assert len(conflicts.get()) == 2
        conflict = conflicts.get()[0]

        assert conflict.is_resolved is False
        assert conflict.dimension.name == "/y"
        assert isinstance(conflict, ChangedDimensionConflict)

    def test_changed_dim_userconfig_conflict(
        self, parent_config, changed_userconfig_config
    ):
        """Test if changed dimension from user's config is currently detected"""
        conflicts = detect_conflicts(parent_config, changed_userconfig_config)

        assert len(conflicts.get()) == 4
        conflict = conflicts.get()[0]

        assert conflict.is_resolved is False
        assert conflict.dimension.name == "/b"
        assert isinstance(conflict, ChangedDimensionConflict)

    def test_algo_conflict(self, parent_config, changed_algo_config):
        """Test if algorithm changes are currently detected"""
        conflicts = detect_conflicts(parent_config, changed_algo_config)

        assert len(conflicts.get()) == 2
        conflict = conflicts.get()[0]

        assert conflict.is_resolved is False
        assert conflict.old_config["algorithms"] == "random"
        assert conflict.new_config["algorithms"] == "stupid-grid"
        assert isinstance(conflict, AlgorithmConflict)

    def test_orion_version_conflict(self, parent_config, changed_orion_version_config):
        """Test if orion version changes are currently detected"""
        conflicts = detect_conflicts(parent_config, changed_orion_version_config)

        assert len(conflicts.get()) == 2
        conflict = conflicts.get()[1]

        assert conflict.is_resolved is False
        assert conflict.old_config["metadata"]["orion_version"] == "XYZ"
        assert conflict.new_config["metadata"]["orion_version"] == "UVW"
        assert isinstance(conflict, OrionVersionConflict)

    def test_code_conflict(self, parent_config, changed_code_config):
        """Test if code commit hash change is currently detected"""
        conflicts = detect_conflicts(parent_config, changed_code_config)

        assert len(conflicts.get()) == 2
        conflict = conflicts.get()[0]

        assert conflict.is_resolved is False
        assert conflict.old_config["metadata"]["VCS"]["HEAD_sha"] == "test"
        assert conflict.new_config["metadata"]["VCS"]["HEAD_sha"] == "new_test"
        assert isinstance(conflict, CodeConflict)

    def test_ignore_code_conflict(self, parent_config, changed_code_config):
        """Test if ignored code commit hash change is detected as a conflict"""
        conflicts = detect_conflicts(
            parent_config, changed_code_config, {"ignore_code_changes": True}
        )

        assert len(conflicts.get()) == 1

    def test_config_new_name_no_conflict(self, parent_config, same_userconfig_config):
        """Test if same configuration file with a different name is not detected as a conflict"""
        conflicts = detect_conflicts(parent_config, same_userconfig_config)

        assert parent_config["metadata"]["user_args"][-1].startswith("--config=")
        assert same_userconfig_config["metadata"]["user_args"][-1].startswith(
            "--config="
        )
        assert (
            parent_config["metadata"]["user_args"][-1]
            != same_userconfig_config["metadata"]["user_args"][-1]
        )

        assert len(conflicts.get()) == 2
        assert not conflicts.get([ExperimentNameConflict])[0].is_resolved

    def test_config_non_dim_conflict(self, parent_config, changed_userconfig_config):
        """Test if changed configuration file is detected as a conflict"""
        conflicts = detect_conflicts(parent_config, changed_userconfig_config)

        assert len(conflicts.get()) == 4
        assert not conflicts.get([ChangedDimensionConflict])[0].is_resolved
        assert not conflicts.get([ExperimentNameConflict])[0].is_resolved
        assert not conflicts.get([ScriptConfigConflict])[0].is_resolved

    def test_cli_conflict(self, parent_config, changed_cli_config):
        """Test if changed command line call is detected as a conflict"""
        conflicts = detect_conflicts(parent_config, changed_cli_config)

        assert len(conflicts.get()) == 2

        assert not conflicts.get([ExperimentNameConflict])[0].is_resolved
        assert not conflicts.get([CommandLineConflict])[0].is_resolved

    def test_cli_ignored_conflict(self, parent_config, changed_cli_config):
        """Test if ignored changed command line call is detected as a conflict"""
        changed_cli_config["metadata"]["user_args"].pop()
        conflicts = detect_conflicts(
            parent_config,
            changed_cli_config,
            {"non_monitored_arguments": ["u", "another"]},
        )

        assert len(conflicts.get()) == 1

        assert not conflicts.get([ExperimentNameConflict])[0].is_resolved


class TestResolutions:
    """Test resolution of conflicts"""

    def test_add_single_hit(self, storage, parent_config, new_config_with_w):
        """Test if adding a dimension only touches the correct status"""
        del new_config_with_w["metadata"]["user_args"][2]
        backward.populate_space(new_config_with_w)
        conflicts = detect_conflicts(parent_config, new_config_with_w)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )
        branch_builder.add_dimension("w_d")

        assert len(conflicts.get()) == 3
        assert conflicts.get([ExperimentNameConflict])[0].is_resolved
        assert conflicts.get([NewDimensionConflict])[0].is_resolved
        assert not conflicts.get([MissingDimensionConflict])[0].is_resolved

    def test_add_new(self, parent_config, new_config_with_w, storage):
        """Test if adding a new dimension solves the conflict"""
        conflicts = detect_conflicts(parent_config, new_config_with_w)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )
        branch_builder.add_dimension("w_d")

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.AddDimensionResolution)

    def test_add_changed(self, parent_config, changed_config, storage):
        """Test if adding a changed dimension solves the conflict"""
        conflicts = detect_conflicts(parent_config, changed_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )
        branch_builder.add_dimension("y")

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.ChangeDimensionResolution)

    def test_remove_missing(self, parent_config, missing_config, storage):
        """Test if removing a missing dimension solves the conflict"""
        conflicts = detect_conflicts(parent_config, missing_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )
        branch_builder.remove_dimension("x")

        assert len(conflicts.get()) == 3
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.RemoveDimensionResolution)

    def test_rename_missing(self, parent_config, missing_config, storage):
        """Test if renaming a dimension to another solves both conflicts"""
        missing_config["metadata"]["user_args"].append("-w_d~uniform(0,1)")
        backward.populate_space(missing_config)
        conflicts = detect_conflicts(parent_config, missing_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )
        branch_builder.rename_dimension("x", "w_d")

        assert len(conflicts.get()) == 4

        assert conflicts.get([ExperimentNameConflict])[0].is_resolved
        assert conflicts.get([NewDimensionConflict])[0].is_resolved
        assert conflicts.get([MissingDimensionConflict])[0].is_resolved
        assert not conflicts.get([MissingDimensionConflict])[1].is_resolved

        resolved_conflicts = conflicts.get_resolved()
        assert len(resolved_conflicts) == 3
        assert resolved_conflicts[1].resolution is resolved_conflicts[2].resolution
        assert isinstance(
            resolved_conflicts[1].resolution,
            resolved_conflicts[1].RenameDimensionResolution,
        )
        assert resolved_conflicts[1].resolution.conflict.dimension.name == "/x"
        assert (
            resolved_conflicts[1].resolution.new_dimension_conflict.dimension.name
            == "/w_d"
        )

    def test_rename_missing_changed(self, parent_config, missing_config, storage):
        """Test if renaming a dimension to another with different prior solves both conflicts but
        creates a new one which is not solved
        """
        missing_config["metadata"]["user_args"].append("-w_d~normal(0,1)")
        backward.populate_space(missing_config)
        conflicts = detect_conflicts(parent_config, missing_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        assert len(conflicts.get()) == 4

        branch_builder.rename_dimension("x", "w_d")

        assert len(conflicts.get()) == 5

        assert conflicts.get([ExperimentNameConflict])[0].is_resolved
        assert conflicts.get([NewDimensionConflict])[0].is_resolved
        assert conflicts.get([MissingDimensionConflict])[0].is_resolved
        assert not conflicts.get([MissingDimensionConflict])[1].is_resolved
        assert not conflicts.get([ChangedDimensionConflict])[0].is_resolved

        resolved_conflicts = conflicts.get_resolved()
        assert len(resolved_conflicts) == 3
        assert resolved_conflicts[1].resolution is resolved_conflicts[2].resolution
        assert isinstance(
            resolved_conflicts[1].resolution,
            resolved_conflicts[1].RenameDimensionResolution,
        )
        assert resolved_conflicts[1].resolution.conflict.dimension.name == "/x"
        assert (
            resolved_conflicts[1].resolution.new_dimension_conflict.dimension.name
            == "/w_d"
        )

    def test_reset_dimension(self, parent_config, new_config_with_w, storage):
        """Test if resetting a dimension unsolves the conflict"""
        conflicts = detect_conflicts(parent_config, new_config_with_w)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        branch_builder.add_dimension("w_d")
        assert len(conflicts.get_resolved()) == 2

        with pytest.raises(ValueError) as exc:
            branch_builder.reset("w_d~+")
        assert "'w_d~+' is not in list" in str(exc.value)
        assert len(conflicts.get_resolved()) == 2

        branch_builder.reset("w_d~+normal(0, 1)")

        assert len(conflicts.get()) == 2

        conflict = conflicts.get(dimension_name="w_d")[0]

        assert not conflict.is_resolved
        assert isinstance(conflict, NewDimensionConflict)
        assert len(conflicts.get_resolved()) == 1

    def test_name_experiment(
        self, bad_exp_parent_config, bad_exp_child_config, storage
    ):
        """Test if having the same experiment name does not create a conflict."""
        backward.populate_space(bad_exp_parent_config)
        backward.populate_space(bad_exp_child_config)
        storage.create_experiment(bad_exp_parent_config)
        storage.create_experiment(bad_exp_child_config)
        conflicts = detect_conflicts(bad_exp_parent_config, bad_exp_parent_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        assert len(conflicts.get()) == 1
        assert len(conflicts.get_resolved()) == 0

        conflict = conflicts.get([ExperimentNameConflict])[0]

        assert conflict.new_config["name"] == "test"
        assert not conflict.is_resolved
        branch_builder.change_experiment_name("test2")
        assert len(conflicts.get_resolved()) == 1
        assert conflict.new_config["name"] == "test2"
        assert conflict.is_resolved

    def test_bad_name_experiment(
        self, parent_config, child_config, monkeypatch, storage
    ):
        """Test if changing the experiment names does not work for invalid name and revert
        to old one
        """

        def _is_unique(self, *args, **kwargs):
            return False

        def _versions(self, *args, **kwargs):
            return True

        monkeypatch.setattr(
            ExperimentNameConflict.ExperimentNameResolution,
            "_name_is_unique",
            _is_unique,
        )
        monkeypatch.setattr(
            ExperimentNameConflict.ExperimentNameResolution,
            "_check_for_greater_versions",
            _versions,
        )

        conflicts = detect_conflicts(parent_config, child_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        assert len(conflicts.get()) == 1
        assert len(conflicts.get_resolved()) == 0

        conflict = conflicts.get([ExperimentNameConflict])[0]

        assert not conflict.is_resolved
        branch_builder.change_experiment_name("test2")
        assert len(conflicts.get_resolved()) == 0
        assert conflict.new_config["name"] == "test"
        assert not conflict.is_resolved

    def test_algo_change(self, parent_config, changed_algo_config, storage):
        """Test if setting the algorithm conflict solves it"""
        conflicts = detect_conflicts(parent_config, changed_algo_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1

        branch_builder.set_algo()

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.AlgorithmResolution)

    def test_orion_version_change(
        self, parent_config, changed_orion_version_config, storage
    ):
        """Test if setting the orion version conflict solves it"""
        conflicts = detect_conflicts(parent_config, changed_orion_version_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1

        branch_builder.set_orion_version()

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.OrionVersionResolution)

    def test_code_change(self, parent_config, changed_code_config, storage):
        """Test if giving a proper change-type solves the code conflict"""
        conflicts = detect_conflicts(parent_config, changed_code_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1

        branch_builder.set_code_change_type(evc.adapters.CodeChange.types[0])

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]
        assert conflict.is_resolved
        assert isinstance(conflict, CodeConflict)

    def test_bad_code_change(self, capsys, parent_config, changed_code_config, storage):
        """Test if giving an invalid change-type prints error message and do nothing"""
        conflicts = detect_conflicts(parent_config, changed_code_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )
        capsys.readouterr()
        branch_builder.set_code_change_type("bad-type")
        out, err = capsys.readouterr()
        assert "Invalid code change type" in out.split("\n")[-3]

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1

    def test_config_change(self, parent_config, changed_userconfig_config, storage):
        """Test if giving a proper change-type solves the user script config conflict"""
        conflicts = detect_conflicts(parent_config, changed_userconfig_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        assert len(conflicts.get()) == 4
        assert len(conflicts.get_resolved()) == 1

        branch_builder.set_script_config_change_type(
            evc.adapters.ScriptConfigChange.types[0]
        )

        assert len(conflicts.get()) == 4
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]
        assert conflict.is_resolved
        assert isinstance(conflict, ScriptConfigConflict)

    def test_bad_config_change(
        self, capsys, parent_config, changed_userconfig_config, storage
    ):
        """Test if giving an invalid change-type prints error message and do nothing"""
        conflicts = detect_conflicts(parent_config, changed_userconfig_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )
        capsys.readouterr()
        branch_builder.set_script_config_change_type("bad-type")
        out, err = capsys.readouterr()
        assert "Invalid script's config change type" in out.split("\n")[-3]

        assert len(conflicts.get()) == 4
        assert len(conflicts.get_resolved()) == 1

    def test_cli_change(self, parent_config, changed_cli_config, storage):
        """Test if giving a proper change-type solves the command line conflict"""
        conflicts = detect_conflicts(parent_config, changed_cli_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1

        branch_builder.set_cli_change_type(evc.adapters.CommandLineChange.types[0])

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]
        assert conflict.is_resolved
        assert isinstance(conflict, CommandLineConflict)

    def test_bad_cli_change(self, capsys, parent_config, changed_cli_config, storage):
        """Test if giving an invalid change-type prints error message and do nothing"""
        conflicts = detect_conflicts(parent_config, changed_cli_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )
        capsys.readouterr()
        branch_builder.set_cli_change_type("bad-type")
        out, err = capsys.readouterr()
        assert "Invalid cli change type" in out.split("\n")[-3]

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1

    def test_solve_all_automatically(self, conflicts, storage):
        """Test if all conflicts all automatically resolve by the ExperimentBranchBuilder."""
        ExperimentBranchBuilder(conflicts, storage=storage)

        assert len(conflicts.get_resolved()) == 9


class TestResolutionsWithMarkers:
    """Test resolution of conflicts with markers"""

    def test_add_new(self, parent_config, new_config_with_w, storage):
        """Test if new dimension conflict is automatically resolved"""
        new_config_with_w["metadata"]["user_args"][-1] = "-w_d~+normal(0,1)"
        conflicts = detect_conflicts(parent_config, new_config_with_w)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.AddDimensionResolution)

    def test_add_new_default(self, parent_config, new_config_with_w, storage):
        """Test if new dimension conflict is automatically resolved"""
        new_config_with_w["metadata"]["user_args"][
            -1
        ] = "-w_d~+normal(0,1,default_value=0)"
        backward.populate_space(new_config_with_w)
        conflicts = detect_conflicts(parent_config, new_config_with_w)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.AddDimensionResolution)
        assert conflict.resolution.default_value == 0.0

    def test_add_bad_default(self, parent_config, new_config_with_w):
        """Test if new dimension conflict raises an error if marked with invalid default value"""
        new_config_with_w["metadata"]["user_args"][
            -1
        ] = "-w_d~+normal(0,1,default_value='a')"
        backward.populate_space(new_config_with_w)
        with pytest.raises(TypeError) as exc:
            detect_conflicts(parent_config, new_config_with_w)
        assert "Parameter '/w_d': Incorrect arguments." in str(exc.value)

    def test_add_changed(self, parent_config, changed_config, storage):
        """Test if changed dimension conflict is automatically resolved"""
        changed_config["metadata"]["user_args"][3] = changed_config["metadata"][
            "user_args"
        ][3].replace("~", "~+")
        conflicts = detect_conflicts(parent_config, changed_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.ChangeDimensionResolution)

    def test_remove_missing(self, parent_config, child_config, storage):
        """Test if missing dimension conflict is automatically resolved"""
        child_config["metadata"]["user_args"][2] = "-x~-"
        backward.populate_space(child_config)
        conflicts = detect_conflicts(parent_config, child_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.RemoveDimensionResolution)

    def test_remove_missing_default(self, parent_config, child_config, storage):
        """Test if missing dimension conflict is automatically resolved"""
        child_config["metadata"]["user_args"][2] = "-x~-0.5"
        backward.populate_space(child_config)
        conflicts = detect_conflicts(parent_config, child_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.RemoveDimensionResolution)
        assert conflict.resolution.default_value == 0.5

    def test_remove_missing_bad_default(self, parent_config, child_config, storage):
        """Test if missing dimension conflict raises an error if marked with invalid default"""
        child_config["metadata"]["user_args"][2] = "-x~--100"
        backward.populate_space(child_config)
        conflicts = detect_conflicts(parent_config, child_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1

        conflict = conflicts.get()[1]

        assert not conflict.is_resolved
        assert isinstance(conflict, MissingDimensionConflict)

    def test_rename_missing(self, parent_config, child_config, storage):
        """Test if renaming is automatically applied with both conflicts resolved"""
        child_config["metadata"]["user_args"].append("-w_a~uniform(0,1)")
        child_config["metadata"]["user_args"].append("-w_b~normal(0,1)")
        child_config["metadata"]["user_args"][2] = "-x~>w_a"
        backward.populate_space(child_config)
        conflicts = detect_conflicts(parent_config, child_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 4

        assert conflicts.get([ExperimentNameConflict])[0].is_resolved
        assert conflicts.get(dimension_name="x")[0].is_resolved
        assert conflicts.get(dimension_name="w_a")[0].is_resolved
        assert not conflicts.get(dimension_name="w_b")[0].is_resolved

        resolved_conflicts = conflicts.get_resolved()
        assert len(resolved_conflicts) == 3
        assert resolved_conflicts[1].resolution is resolved_conflicts[2].resolution
        assert isinstance(
            resolved_conflicts[1].resolution,
            resolved_conflicts[1].RenameDimensionResolution,
        )
        assert resolved_conflicts[1].resolution.conflict.dimension.name == "/x"
        assert (
            resolved_conflicts[1].resolution.new_dimension_conflict.dimension.name
            == "/w_a"
        )

    def test_rename_invalid(self, parent_config, child_config, storage):
        """Test if renaming to invalid dimension raises an error"""
        child_config["metadata"]["user_args"].append("-w_a~uniform(0,1)")
        child_config["metadata"]["user_args"].append("-w_b~uniform(0,1)")
        child_config["metadata"]["user_args"][2] = "-x~>w_c"
        backward.populate_space(child_config)
        conflicts = detect_conflicts(parent_config, child_config)
        with pytest.raises(ValueError) as exc:
            ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)
        assert "Dimension name 'w_c' not found in conflicts" in str(exc.value)

    def test_rename_missing_changed(self, parent_config, child_config, storage):
        """Test if renaming is automatically applied with both conflicts resolved,
        but not the new one because of prior change
        """
        child_config["metadata"]["user_args"].append("-w_a~uniform(0,1)")
        child_config["metadata"]["user_args"].append("-w_b~normal(0,1)")
        child_config["metadata"]["user_args"][2] = "-x~>w_b"
        backward.populate_space(child_config)
        conflicts = detect_conflicts(parent_config, child_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 5

        assert conflicts.get([ExperimentNameConflict])[0].is_resolved
        assert conflicts.get(dimension_name="x")[0].is_resolved
        assert conflicts.get([NewDimensionConflict], dimension_name="w_b")[
            0
        ].is_resolved
        assert not conflicts.get([ChangedDimensionConflict], dimension_name="w_b")[
            0
        ].is_resolved
        assert not conflicts.get(dimension_name="w_a")[0].is_resolved

        resolved_conflicts = conflicts.get_resolved()
        assert len(resolved_conflicts) == 3
        assert resolved_conflicts[1].resolution is resolved_conflicts[2].resolution
        assert isinstance(
            resolved_conflicts[1].resolution,
            resolved_conflicts[1].RenameDimensionResolution,
        )
        assert resolved_conflicts[1].resolution.conflict.dimension.name == "/x"
        assert (
            resolved_conflicts[1].resolution.new_dimension_conflict.dimension.name
            == "/w_b"
        )

    def test_rename_missing_changed_marked(self, parent_config, child_config, storage):
        """Test if renaming is automatically applied with all conflicts resolved including
        the new one caused by prior change
        """
        child_config["metadata"]["user_args"].append("-w_a~uniform(0,1)")
        child_config["metadata"]["user_args"].append("-w_b~+normal(0,1)")
        child_config["metadata"]["user_args"][2] = "-x~>w_b"
        backward.populate_space(child_config)
        conflicts = detect_conflicts(parent_config, child_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 5

        assert conflicts.get([ExperimentNameConflict])[0].is_resolved
        assert conflicts.get(dimension_name="x")[0].is_resolved
        assert conflicts.get([NewDimensionConflict], dimension_name="w_b")[
            0
        ].is_resolved
        assert conflicts.get([ChangedDimensionConflict], dimension_name="w_b")[
            0
        ].is_resolved
        assert not conflicts.get(dimension_name="w_a")[0].is_resolved

        resolved_conflicts = conflicts.get_resolved()
        assert len(resolved_conflicts) == 4
        assert resolved_conflicts[1].resolution is resolved_conflicts[2].resolution
        assert isinstance(
            resolved_conflicts[1].resolution,
            resolved_conflicts[1].RenameDimensionResolution,
        )
        assert resolved_conflicts[1].resolution.conflict.dimension.name == "/x"
        assert (
            resolved_conflicts[1].resolution.new_dimension_conflict.dimension.name
            == "/w_b"
        )

    def test_name_experiment_version_update(self, parent_config, child_config, storage):
        """Test if experiment name conflict is automatically resolved with version update"""
        old_name = "test"
        new_version = 2
        storage.create_experiment(parent_config)
        child_config["version"] = 1
        conflicts = detect_conflicts(parent_config, child_config)
        ExperimentBranchBuilder(conflicts, storage=storage)

        assert len(conflicts.get()) == 1
        assert len(conflicts.get_resolved()) == 1

        conflict = conflicts.get()[0]

        assert conflict.resolution.new_name == old_name
        assert conflict.resolution.new_version == 2
        assert conflict.new_config["name"] == old_name
        assert conflict.new_config["version"] == new_version
        assert conflict.is_resolved

    def test_name_experiment_name_change(self, parent_config, child_config, storage):
        """Test if experiment name conflict is automatically resolved when new name provided"""
        new_name = "test2"
        storage.create_experiment(parent_config)
        storage.create_experiment(child_config)
        child_config2 = copy.deepcopy(child_config)
        child_config2["version"] = 1
        conflicts = detect_conflicts(parent_config, child_config2)
        ExperimentBranchBuilder(conflicts, branch_to=new_name, storage=storage)

        assert len(conflicts.get()) == 1
        assert len(conflicts.get_resolved()) == 1

        conflict = conflicts.get()[0]

        assert conflict.resolution.new_name == new_name
        assert conflict.resolution.new_version == 1
        assert conflict.new_config["name"] == new_name
        assert conflict.new_config["version"] == 1
        assert conflict.is_resolved

    def test_bad_name_experiment(
        self, parent_config, child_config, monkeypatch, storage
    ):
        """Test if experiment name conflict is not resolved when invalid name is marked"""

        def _is_unique(self, *args, **kwargs):
            return False

        monkeypatch.setattr(
            ExperimentNameConflict.ExperimentNameResolution,
            "_name_is_unique",
            _is_unique,
        )

        conflicts = detect_conflicts(parent_config, child_config)
        ExperimentBranchBuilder(conflicts, branch_to="test2", storage=storage)

        assert len(conflicts.get()) == 1
        assert len(conflicts.get_resolved()) == 0

    def test_code_change(self, parent_config, changed_code_config, storage):
        """Test if code conflict is resolved automatically"""
        change_type = evc.adapters.CodeChange.types[0]
        changed_code_config["code_change_type"] = change_type
        conflicts = detect_conflicts(parent_config, changed_code_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.CodeResolution)
        assert conflict.resolution.type == change_type

    def test_algo_change(self, parent_config, changed_algo_config, storage):
        """Test if algorithm conflict is resolved automatically"""
        changed_algo_config["algorithm_change"] = True
        conflicts = detect_conflicts(parent_config, changed_algo_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.AlgorithmResolution)

    def test_orion_version_change(
        self, parent_config, changed_orion_version_config, storage
    ):
        """Test if orion version conflict is resolved automatically"""
        changed_orion_version_config["orion_version_change"] = True
        conflicts = detect_conflicts(parent_config, changed_orion_version_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.OrionVersionResolution)

    def test_config_change(self, parent_config, changed_userconfig_config, storage):
        """Test if user's script's config conflict is resolved automatically"""
        change_type = evc.adapters.ScriptConfigChange.types[0]
        changed_userconfig_config["config_change_type"] = change_type
        conflicts = detect_conflicts(parent_config, changed_userconfig_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 4
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[1]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.ScriptConfigResolution)
        assert conflict.resolution.type == change_type

    def test_cli_change(self, parent_config, changed_cli_config, storage):
        """Test if command line conflict is resolved automatically"""
        change_type = evc.adapters.CommandLineChange.types[0]
        changed_cli_config["cli_change_type"] = change_type
        conflicts = detect_conflicts(parent_config, changed_cli_config)
        ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]

        assert conflict.is_resolved
        assert isinstance(conflict.resolution, conflict.CommandLineResolution)
        assert conflict.resolution.type == change_type


class TestAdapters:
    """Test creation of adapters"""

    def test_adapter_add_new(self, parent_config, cl_config, storage):
        """Test if a DimensionAddition is created when solving a new conflict"""
        cl_config["metadata"]["user_args"] = ["-w_d~+normal(0,1)"]

        conflicts = detect_conflicts(parent_config, cl_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        adapters = branch_builder.create_adapters().adapters

        assert len(conflicts.get_resolved()) == 2
        assert len(adapters) == 1
        assert isinstance(adapters[0], evc.adapters.DimensionAddition)

    def test_adapter_add_changed(self, parent_config, cl_config, storage):
        """Test if a DimensionPriorChange is created when solving a new conflict"""
        cl_config["metadata"]["user_args"] = ["-y~+uniform(0,1)"]

        conflicts = detect_conflicts(parent_config, cl_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        adapters = branch_builder.create_adapters().adapters

        assert len(conflicts.get_resolved()) == 2
        assert len(adapters) == 1
        assert isinstance(adapters[0], evc.adapters.DimensionPriorChange)

    def test_adapter_remove_missing(self, parent_config, cl_config, storage):
        """Test if a DimensionDeletion is created when solving a new conflict"""
        cl_config["metadata"]["user_args"] = ["-z~-"]

        conflicts = detect_conflicts(parent_config, cl_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        adapters = branch_builder.create_adapters().adapters

        assert len(conflicts.get_resolved()) == 2
        assert len(adapters) == 1
        assert isinstance(adapters[0], evc.adapters.DimensionDeletion)

    def test_adapter_rename_missing(self, parent_config, cl_config, storage):
        """Test if a DimensionRenaming is created when solving a new conflict"""
        cl_config["metadata"]["user_args"] = ["-x~>w_d", "-w_d~+uniform(0,1)"]
        backward.populate_space(cl_config)

        conflicts = detect_conflicts(parent_config, cl_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        adapters = branch_builder.create_adapters().adapters

        assert len(conflicts.get_resolved()) == 3
        assert len(adapters) == 1
        assert isinstance(adapters[0], evc.adapters.DimensionRenaming)

    def test_adapter_rename_different_prior(self, parent_config, cl_config, storage):
        """Test if a DimensionRenaming is created when solving a new conflict"""
        cl_config["metadata"]["user_args"] = ["-x~>w_d", "-w_d~+normal(0,1)"]

        conflicts = detect_conflicts(parent_config, cl_config)
        branch_builder = ExperimentBranchBuilder(
            conflicts, manual_resolution=True, storage=storage
        )

        adapters = branch_builder.create_adapters().adapters

        assert len(conflicts.get_resolved()) == 4
        assert len(adapters) == 2
        assert isinstance(adapters[0], evc.adapters.DimensionRenaming)
        assert isinstance(adapters[1], evc.adapters.DimensionPriorChange)


class TestResolutionsConfig:
    """Test auto-resolution with specific types from orion.core.config.evc"""

    def test_cli_change(self, parent_config, changed_cli_config, storage):
        """Test if giving a proper change-type solves the command line conflict"""
        conflicts = detect_conflicts(parent_config, changed_cli_config)
        orion.core.config.evc.cli_change_type = "noeffect"
        ExperimentBranchBuilder(conflicts, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]
        assert conflict.is_resolved
        assert isinstance(conflict, CommandLineConflict)
        assert conflict.resolution.type == "noeffect"
        orion.core.config.evc.cli_change_type = "break"

    def test_bad_cli_change(self, capsys, parent_config, changed_cli_config, storage):
        """Test if giving an invalid change-type fails the the resolution"""
        conflicts = detect_conflicts(parent_config, changed_cli_config)
        orion.core.config.evc.cli_change_type = "bad-type"
        ExperimentBranchBuilder(conflicts, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1
        orion.core.config.evc.cli_change_type = "break"

    def test_code_change(self, parent_config, changed_code_config, storage):
        """Test if giving a proper change-type solves the code conflict"""
        conflicts = detect_conflicts(parent_config, changed_code_config)
        orion.core.config.evc.code_change_type = "noeffect"
        ExperimentBranchBuilder(conflicts, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 2

        conflict = conflicts.get_resolved()[0]
        assert conflict.is_resolved
        assert isinstance(conflict, CodeConflict)
        assert conflict.resolution.type == "noeffect"
        orion.core.config.evc.code_change_type = "break"

    def test_bad_code_change(self, capsys, parent_config, changed_code_config, storage):
        """Test if giving an invalid change-type prints error message and do nothing"""
        conflicts = detect_conflicts(parent_config, changed_code_config)
        orion.core.config.evc.code_change_type = "bad-type"
        ExperimentBranchBuilder(conflicts, storage=storage)

        assert len(conflicts.get()) == 2
        assert len(conflicts.get_resolved()) == 1
        orion.core.config.evc.code_change_type = "break"

    def test_config_change(self, parent_config, changed_userconfig_config, storage):
        """Test if giving a proper change-type solves the user script config conflict"""
        conflicts = detect_conflicts(parent_config, changed_userconfig_config)
        orion.core.config.evc.config_change_type = "noeffect"
        ExperimentBranchBuilder(conflicts, storage=storage)

        assert len(conflicts.get()) == 4
        assert len(conflicts.get_resolved()) == 4

        conflict = conflicts.get_resolved()[3]
        assert conflict.is_resolved
        assert isinstance(conflict, ScriptConfigConflict)
        assert conflict.resolution.type == "noeffect"

    def test_bad_config_change(
        self, capsys, parent_config, changed_userconfig_config, storage
    ):
        """Test if giving an invalid change-type prints error message and do nothing"""
        conflicts = detect_conflicts(parent_config, changed_userconfig_config)
        orion.core.config.evc.config_change_type = "bad-type"
        ExperimentBranchBuilder(conflicts, storage=storage)

        assert len(conflicts.get()) == 4
        assert len(conflicts.get_resolved()) == 3
        orion.core.config.evc.config_change_type = "break"
