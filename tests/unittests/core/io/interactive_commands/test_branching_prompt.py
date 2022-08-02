#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.io.interactive_commands.branching_prompt`."""
import shlex

import pytest

from orion.core import evc
from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder
from orion.core.io.interactive_commands.branching_prompt import BranchingPrompt
from orion.core.io.space_builder import DimensionBuilder


@pytest.fixture
def new_cat_dimension_conflict(old_config, new_config):
    """Generate a new dimension conflict with categorical prior for new experiment configuration"""
    name = "new-cat"
    prior = 'choices(["hello", 2])'
    dimension = DimensionBuilder().build(name, prior)
    return evc.conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def missing_conflict_with_identical_prior(
    old_config, new_config, new_dimension_conflict
):
    """Generate a missing dimension conflict which have the same prior as the new dim conflict"""
    name = "missing-idem"
    prior = new_dimension_conflict.prior
    dimension = DimensionBuilder().build(name, prior)
    return evc.conflicts.MissingDimensionConflict(
        old_config, new_config, dimension, prior
    )


@pytest.fixture
def missing_cat_dimension_conflict(old_config, new_config):
    """Generate a missing dimension conflict with categorical prior for new experiment
    configuration
    """
    name = "missing-cat"
    prior = 'choices(["goodbye", 5])'
    dimension = DimensionBuilder().build(name, prior)
    return evc.conflicts.MissingDimensionConflict(
        old_config, new_config, dimension, prior
    )


@pytest.fixture
def conflicts(
    new_dimension_conflict,
    new_cat_dimension_conflict,
    changed_dimension_conflict,
    missing_dimension_conflict,
    missing_cat_dimension_conflict,
    missing_conflict_with_identical_prior,
    algorithm_conflict,
    orion_version_conflict,
    code_conflict,
    cli_conflict,
    config_conflict,
    experiment_name_conflict,
):
    """Create a container for conflicts with one of each types for testing purposes"""
    conflicts = evc.conflicts.Conflicts()
    conflicts.register(new_dimension_conflict)
    conflicts.register(new_cat_dimension_conflict)
    conflicts.register(changed_dimension_conflict)
    conflicts.register(missing_dimension_conflict)
    conflicts.register(missing_cat_dimension_conflict)
    conflicts.register(missing_conflict_with_identical_prior)
    conflicts.register(algorithm_conflict)
    conflicts.register(orion_version_conflict)
    conflicts.register(code_conflict)
    conflicts.register(config_conflict)
    conflicts.register(cli_conflict)
    conflicts.register(experiment_name_conflict)
    return conflicts


@pytest.fixture
def branch_builder(storage, conflicts):
    """Generate the experiment branch builder"""
    return ExperimentBranchBuilder(conflicts, manual_resolution=True, storage=storage)


@pytest.fixture
def branch_solver_prompt(branch_builder):
    """Generate the branch solved prompt"""
    return BranchingPrompt(branch_builder)


class TestCommands:
    """Test the commands of the prompt"""

    def test_add_dim(self, conflicts, branch_solver_prompt):
        """Verify that dimension is added properly"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new")
        assert len(conflicts.get_resolved()) == 1

    def test_add_bad_dim(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given when dimension does not exist"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("idontexist")
        out, err = capsys.readouterr()
        assert "Dimension name 'idontexist' not found in conflicts" in out
        assert len(conflicts.get_resolved()) == 0

    def test_add_dim_twice(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_add("new")
        out, err = capsys.readouterr()
        assert "Dimension name 'new' not found in conflicts" in out
        assert len(conflicts.get_resolved()) == 1

    def test_add_dim_with_default(self, conflicts, branch_solver_prompt):
        """Verify that dimension is added with given default value"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new --default-value=9")
        assert len(conflicts.get_resolved()) == 1
        assert conflicts.get_resolved()[0].resolution.default_value == 9

    def test_add_dim_with_bad_default(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given for bad default value"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new --default-value='bad'")
        out, err = capsys.readouterr()
        assert "could not convert string to float: 'bad'" in out
        assert len(conflicts.get_resolved()) == 0

    def test_add_dim_with_cat(self, conflicts, branch_solver_prompt):
        """Verify that categorical dimensions is added properly"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new_cat")
        assert len(conflicts.get_resolved()) == 1

    def test_add_dim_with_cat_default(self, conflicts, branch_solver_prompt):
        """Verify that string and non-string default value are properly parsed"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new_cat --default-value='hello'")
        assert len(conflicts.get_resolved()) == 1
        assert conflicts.get_resolved()[0].resolution.default_value == "hello"

        conflicts.revert(conflicts.get_resolved()[0].resolution)
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new_cat --default-value=2")
        assert len(conflicts.get_resolved()) == 1
        assert conflicts.get_resolved()[0].resolution.default_value == 2

    def test_add_dim_with_cat_bad_default(
        self, capsys, conflicts, branch_solver_prompt
    ):
        """Verify that error message is given for default value of invalid category"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new_cat --default-value='bad'")
        out, err = capsys.readouterr()
        assert "Invalid category: bad" in out
        assert len(conflicts.get_resolved()) == 0

    def test_reset_add(self, conflicts, branch_solver_prompt):
        """Verify that addition resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("new")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_change_dim(self, conflicts, branch_solver_prompt):
        """Verify that changed resolution is created"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("changed")
        assert len(conflicts.get_resolved()) == 1

    def test_change_dim_twice(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("changed")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_add("changed")
        out, err = capsys.readouterr()
        assert "Dimension name 'changed' not found in conflicts" in out
        assert len(conflicts.get_resolved()) == 1

    def test_reset_change(self, conflicts, branch_solver_prompt):
        """Verify that changed resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_add("changed")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_remove_dim(self, conflicts, branch_solver_prompt):
        """Verify that missing dimension is removed"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing")
        assert len(conflicts.get_resolved()) == 1

    def test_remove_bad_dim(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given for non existing dimension"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("idontexist")
        out, err = capsys.readouterr()
        assert "Dimension name 'idontexist' not found in conflicts" in out
        assert len(conflicts.get_resolved()) == 0

    def test_remove_dim_twice(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_remove("missing")
        out, err = capsys.readouterr()
        assert "Dimension name 'missing' not found in conflicts" in out
        assert len(conflicts.get_resolved()) == 1

    def test_remove_dim_with_default(self, conflicts, branch_solver_prompt):
        """Verify that default value is properly parsed for remove resolution"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing --default-value=9")
        assert len(conflicts.get_resolved()) == 1
        assert conflicts.get_resolved()[0].resolution.default_value == 9

    def test_remove_dim_with_bad_default(self, capsys, conflicts, branch_solver_prompt):
        """Verify error message is given when default value is invalid"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing --default-value='bad'")
        out, err = capsys.readouterr()
        assert "could not convert string to float: 'bad'" in out
        assert len(conflicts.get_resolved()) == 0

    def test_remove_dim_with_cat(self, conflicts, branch_solver_prompt):
        """Verify that missing categorical dimension is properly removed"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing_cat")
        assert len(conflicts.get_resolved()) == 1

    def test_remove_dim_with_cat_default(self, conflicts, branch_solver_prompt):
        """Verify that categorical default value is properly parsed for remove resolution"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing_cat --default-value='goodbye'")
        assert len(conflicts.get_resolved()) == 1
        assert conflicts.get_resolved()[0].resolution.default_value == "goodbye"

        conflicts.revert(conflicts.get_resolved()[0].resolution)
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing_cat --default-value=5")
        assert len(conflicts.get_resolved()) == 1
        assert conflicts.get_resolved()[0].resolution.default_value == 5

    def test_remove_dim_with_cat_bad_default(
        self, capsys, conflicts, branch_solver_prompt
    ):
        """Verify that error message is given for default value of invalid category"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing_cat --default-value='bad'")
        out, err = capsys.readouterr()
        assert "Invalid category: bad" in out
        assert len(conflicts.get_resolved()) == 0

    def test_reset_remove(self, conflicts, branch_solver_prompt):
        """Verify that remove resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_remove("missing")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_rename_dim(self, conflicts, branch_solver_prompt):
        """Verify that rename resolution is properly created"""
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_rename("missing_idem new")
        assert len(conflicts.get_resolved()) == 2
        assert len(conflicts.get()) == 12

    def test_rename_and_change_dim(self, conflicts, branch_solver_prompt):
        """Verify that rename resolution is created and a new side-effect conflict added"""
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_rename("missing new")
        assert len(conflicts.get_resolved()) == 2
        assert len(conflicts.get()) == 13

    def test_rename_bad_dim(self, capsys, conflicts, branch_solver_prompt):
        """Verify error messages when attempting invalid renamings"""
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_rename("new_cat new")
        out, err = capsys.readouterr()
        assert "Dimension name 'new_cat' not found in conflicts" in out
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12

        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_rename("missing missing_cat")
        out, err = capsys.readouterr()
        assert "Dimension name 'missing_cat' not found in conflicts" in out
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12

    def test_reset_rename_with_same_priors(self, conflicts, branch_solver_prompt):
        """Verify that rename resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_rename("missing_idem new")
        assert len(conflicts.get_resolved()) == 2
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12

    def test_reset_rename_with_different_priors(self, conflicts, branch_solver_prompt):
        """Verify that rename resolution is reverted and side-effect conflict is removed"""
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_rename("missing new")
        assert len(conflicts.get_resolved()) == 2
        assert len(conflicts.get()) == 13
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12

    def test_set_code_change_type(self, conflicts, branch_solver_prompt):
        """Verify that code change resolution is created"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_code("break")
        assert len(conflicts.get_resolved()) == 1

    def test_set_code_change_bad_type(self, capsys, conflicts, branch_solver_prompt):
        """Verify error message when attempting code resolution with bad type"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_code("bad")
        out, err = capsys.readouterr()
        assert "invalid choice: 'bad'" in err
        assert len(conflicts.get_resolved()) == 0

    def test_set_code_change_type_twice(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_code("break")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_code("noeffect")
        out, err = capsys.readouterr()
        assert "No code conflicts to solve" in out
        assert len(conflicts.get_resolved()) == 1

    def test_reset_code(self, conflicts, branch_solver_prompt):
        """Verify that code resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_code("break")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_set_algo(self, conflicts, branch_solver_prompt):
        """Verify that algo resolution is created"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_algo("")
        assert len(conflicts.get_resolved()) == 1

    def test_set_algo_twice(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_algo("")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_algo("")
        out, err = capsys.readouterr()
        assert "No algo conflict to solve" in out
        assert len(conflicts.get_resolved()) == 1

    def test_reset_algo(self, conflicts, branch_solver_prompt):
        """Verify that algo resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_algo("")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"' {str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_set_orion_version(self, conflicts, branch_solver_prompt):
        """Verify that orion version resolution is created"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_orion_version("")
        assert len(conflicts.get_resolved()) == 1

    def test_set_orion_version_twice(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_orion_version("")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_orion_version("")
        out, err = capsys.readouterr()
        assert "No orion version conflict to solve" in out
        assert len(conflicts.get_resolved()) == 1

    def test_reset_orion_version(self, conflicts, branch_solver_prompt):
        """Verify that orion version resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_orion_version("")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"' {str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_set_config_change_type(self, conflicts, branch_solver_prompt):
        """Verify that script's config resolution is created"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_config("break")
        assert len(conflicts.get_resolved()) == 1

    def test_set_config_change_bad_type(self, capsys, conflicts, branch_solver_prompt):
        """Verify error message when attempting config resolution with bad type"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_config("bad")
        out, err = capsys.readouterr()
        assert "invalid choice: 'bad'" in err
        assert len(conflicts.get_resolved()) == 0

    def test_set_config_change_type_twice(
        self, capsys, conflicts, branch_solver_prompt
    ):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_config("break")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_config("noeffect")
        out, err = capsys.readouterr()
        assert "No script's config conflicts to solve" in out
        assert len(conflicts.get_resolved()) == 1

    def test_reset_config(self, conflicts, branch_solver_prompt):
        """Verify that script's config resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_config("break")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_set_commandline_change_type(self, conflicts, branch_solver_prompt):
        """Verify that cli resolution is created"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_commandline("break")
        assert len(conflicts.get_resolved()) == 1

    def test_set_commandline_change_bad_type(
        self, capsys, conflicts, branch_solver_prompt
    ):
        """Verify error message when attempting cli resolution with bad type"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_commandline("bad")
        out, err = capsys.readouterr()
        assert "invalid choice: 'bad'" in err
        assert len(conflicts.get_resolved()) == 0

    def test_set_commandline_change_type_twice(
        self, capsys, conflicts, branch_solver_prompt
    ):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_commandline("break")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_commandline("noeffect")
        out, err = capsys.readouterr()
        assert "No command line conflicts to solve" in out
        assert len(conflicts.get_resolved()) == 1

    def test_reset_commandline(self, conflicts, branch_solver_prompt):
        """Verify that cli resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_commandline("break")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_set_experiment_name(self, conflicts, branch_solver_prompt):
        """Verify that experiment name resolution is created"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_name("new-name")
        assert len(conflicts.get_resolved()) == 1

    def test_set_experiment_bad_name(self, capsys, conflicts, branch_solver_prompt):
        """Verify error message when attempting experiment name resolution with bad name"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_name("test")
        out, err = capsys.readouterr()
        assert "Experiment name 'test' already exist for version '1'" in out
        assert len(conflicts.get_resolved()) == 0

    def test_set_experiment_name_twice(self, capsys, conflicts, branch_solver_prompt):
        """Verify that error message is given trying to solve twice the same conflict"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_name("new-name")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_name("whatever")
        out, err = capsys.readouterr()
        assert "No experiment name conflict to solve" in out
        assert len(conflicts.get_resolved()) == 1

    def test_reset_exp_name(self, conflicts, branch_solver_prompt):
        """Verify that experiment name resolution is reverted"""
        assert len(conflicts.get_resolved()) == 0
        branch_solver_prompt.do_name("new-name")
        assert len(conflicts.get_resolved()) == 1
        branch_solver_prompt.do_reset(
            f"'{str(conflicts.get_resolved()[0].resolution)}'"
        )
        assert len(conflicts.get_resolved()) == 0

    def test_commit_wont_quit_if_not_solved(self, conflicts, branch_solver_prompt):
        """Verify that commit will not quit if some conflicts are not resolved"""
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_auto("")
        assert len(conflicts.get()) == 12
        assert len(conflicts.get_resolved()) == 11

        assert branch_solver_prompt.do_commit("") is False

    def test_commit_quit_if_solved(self, conflicts, branch_solver_prompt):
        """Verify that commit will quit when all conflicts are resolved"""
        branch_solver_prompt.do_auto("")
        conflicts.conflicts = [conflicts.get_resolved()[0]]
        assert conflicts.are_resolved
        assert branch_solver_prompt.do_commit("") is True

    def test_auto(self, conflicts, branch_solver_prompt):
        """Verify that all conflicts which requires not input are automatically resolved"""
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_auto("")
        assert len(conflicts.get()) == 12
        assert len(conflicts.get_resolved()) == 11

    def test_reset_many(self, conflicts, branch_solver_prompt):
        """Verify that all resolutions are reverted"""
        assert len(conflicts.get_resolved()) == 0
        assert len(conflicts.get()) == 12
        branch_solver_prompt.do_auto("")
        assert len(conflicts.get()) == 12
        assert len(conflicts.get_resolved()) == 11

        reset_strings = []
        for resolution in conflicts.get_resolutions():
            resolution_string = shlex.quote(str(resolution))
            # Otherwise --argument is interpreted as an argument by argparse rather than a
            # positional string
            if not resolution_string.startswith("'"):
                resolution_string = f"' {resolution_string}'"
            reset_strings.append(resolution_string)
        branch_solver_prompt.do_reset(" ".join(reset_strings))
        assert len(conflicts.get_resolved()) == 0
