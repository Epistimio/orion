#!/usr/bin/env python
"""Collection of tests for :mod:`orion.core.evc.conflicts`."""
import pprint

import pytest

import orion.core.utils.backward as backward
from orion.algo.space import Dimension
from orion.core import evc
from orion.core.evc import conflicts as conflict
from orion.testing.evc import add_default_config_for_missing


@pytest.fixture
def conflicts(
    new_dimension_conflict,
    changed_dimension_conflict,
    missing_dimension_conflict,
    algorithm_conflict,
    orion_version_conflict,
    code_conflict,
    experiment_name_conflict,
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
    return conflicts


class TestNewDimensionConflict:
    """Tests methods related to new dimension conflicts"""

    def test_try_resolve_no_default(self, new_dimension_conflict):
        """Verify that resolution is achievable without default value"""
        assert not new_dimension_conflict.is_resolved
        resolution = new_dimension_conflict.try_resolve()
        assert new_dimension_conflict.is_resolved
        assert isinstance(resolution, new_dimension_conflict.AddDimensionResolution)
        assert resolution.default_value is Dimension.NO_DEFAULT_VALUE

    def test_try_resolve_default_from_dim(self, new_dimension_with_default_conflict):
        """Verify that resolution includes dimension's default value if none is provided"""
        assert not new_dimension_with_default_conflict.is_resolved
        resolution = new_dimension_with_default_conflict.try_resolve()
        assert new_dimension_with_default_conflict.is_resolved
        assert isinstance(
            resolution, new_dimension_with_default_conflict.AddDimensionResolution
        )
        assert (
            resolution.default_value
            == new_dimension_with_default_conflict.dimension.default_value
        )

    def test_try_resolve_default(self, new_dimension_conflict):
        """Verify that resolution includes default value if provided"""
        default_value = 1.2
        assert not new_dimension_conflict.is_resolved
        resolution = new_dimension_conflict.try_resolve(default_value)
        assert new_dimension_conflict.is_resolved
        assert isinstance(resolution, new_dimension_conflict.AddDimensionResolution)
        assert resolution.default_value == default_value

    def test_try_resolve_default_over_dim(self, new_dimension_with_default_conflict):
        """Verify that resolution overwrites dimension's default value if one is provided"""
        default_value = 1.2
        assert not new_dimension_with_default_conflict.is_resolved
        resolution = new_dimension_with_default_conflict.try_resolve(default_value)
        assert new_dimension_with_default_conflict.is_resolved
        assert isinstance(
            resolution, new_dimension_with_default_conflict.AddDimensionResolution
        )
        assert resolution.default_value == default_value

    def test_try_resolve_bad_default(self, new_dimension_conflict):
        """Verify that resolution fails if default value is invalid"""
        assert not new_dimension_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            new_dimension_conflict.try_resolve("bad-default")
        assert "could not convert string to float: 'bad-default'" in str(exc.value)

    def test_try_resolve_twice(self, new_dimension_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not new_dimension_conflict.is_resolved
        assert isinstance(
            new_dimension_conflict.try_resolve(),
            new_dimension_conflict.AddDimensionResolution,
        )
        assert new_dimension_conflict.is_resolved
        assert new_dimension_conflict.try_resolve() is None

    def test_repr(self, new_dimension_conflict):
        """Verify the representation of conflict for user interface"""
        assert repr(new_dimension_conflict) == "New new"


class TestChangedDimensionConflict:
    """Tests methods related to changed dimension prior conflicts"""

    def test_try_resolve_twice(self, changed_dimension_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not changed_dimension_conflict.is_resolved
        assert isinstance(
            changed_dimension_conflict.try_resolve(),
            changed_dimension_conflict.ChangeDimensionResolution,
        )
        assert changed_dimension_conflict.is_resolved
        assert changed_dimension_conflict.try_resolve() is None

    def test_try_resolve(self, changed_dimension_conflict):
        """Verify that resolution is achievable without any input"""
        assert not changed_dimension_conflict.is_resolved
        resolution = changed_dimension_conflict.try_resolve()
        assert isinstance(
            resolution, changed_dimension_conflict.ChangeDimensionResolution
        )
        assert changed_dimension_conflict.is_resolved
        assert resolution.conflict is changed_dimension_conflict

    def test_repr(self, changed_dimension_conflict):
        """Verify the representation of conflict for user interface"""
        assert (
            repr(changed_dimension_conflict)
            == "changed~uniform(-10, 10) != changed~normal(0, 2)"
        )


@pytest.mark.parametrize(
    "dimension_conflict",
    [
        pytest.lazy_fixture("missing_dimension_conflict"),
        pytest.lazy_fixture("missing_dimension_from_config_conflict"),
    ],
)
class TestMissingDimensionConflict:
    """Tests methods related to missing dimension conflicts"""

    def test_get_marked_arguments(self, conflicts, dimension_conflict):
        import pprint

        pprint.pprint(dimension_conflict.new_config["metadata"])
        assert dimension_conflict.get_marked_arguments(conflicts) == {}

    def test_try_resolve_no_default(self, dimension_conflict):
        """Verify that resolution is achievable without default value provided"""
        assert not dimension_conflict.is_resolved
        resolution = dimension_conflict.try_resolve()
        assert dimension_conflict.is_resolved
        assert isinstance(resolution, dimension_conflict.RemoveDimensionResolution)
        assert resolution.default_value is Dimension.NO_DEFAULT_VALUE

    def test_try_resolve_default_from_dim(self, dimension_conflict):
        """Verify that resolution uses default value from dim when no provided by user"""
        missing_dimension_with_default_conflict = add_default_config_for_missing(
            dimension_conflict, 0.0
        )
        assert not missing_dimension_with_default_conflict.is_resolved
        resolution = missing_dimension_with_default_conflict.try_resolve()
        assert missing_dimension_with_default_conflict.is_resolved
        assert isinstance(
            resolution,
            missing_dimension_with_default_conflict.RemoveDimensionResolution,
        )
        assert (
            resolution.default_value
            == missing_dimension_with_default_conflict.dimension.default_value
        )

    def test_try_resolve_default(self, dimension_conflict):
        """Verify that resolution uses default value provided by user"""
        default_value = 1.2
        assert not dimension_conflict.is_resolved
        resolution = dimension_conflict.try_resolve(default_value=default_value)
        assert dimension_conflict.is_resolved
        assert isinstance(resolution, dimension_conflict.RemoveDimensionResolution)
        assert resolution.default_value == default_value

    def test_try_resolve_default_over_dim(self, dimension_conflict):
        """Verify that resolution overwrite dimension's default value when user provide one"""
        missing_dimension_with_default_conflict = add_default_config_for_missing(
            dimension_conflict, 0.0
        )
        default_value = 1.2
        assert not missing_dimension_with_default_conflict.is_resolved
        resolution = missing_dimension_with_default_conflict.try_resolve(
            default_value=default_value
        )
        assert missing_dimension_with_default_conflict.is_resolved
        assert isinstance(
            resolution,
            missing_dimension_with_default_conflict.RemoveDimensionResolution,
        )
        assert resolution.default_value == default_value

    def test_try_resolve_bad_default(self, dimension_conflict):
        """Verify that resolution fails if default value is invalid"""
        assert not dimension_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            dimension_conflict.try_resolve(default_value="-100")
        assert "Default value `-100.0` is outside" in str(exc.value)

    def test_try_resolve_renaming(self, dimension_conflict, new_dimension_conflict):
        """Verify that resolution is a renaming when new_dimension_conflict is provided"""
        assert not dimension_conflict.is_resolved
        resolution = dimension_conflict.try_resolve(new_dimension_conflict)
        assert isinstance(resolution, dimension_conflict.RenameDimensionResolution)
        assert resolution.conflict is dimension_conflict
        assert resolution.new_dimension_conflict is new_dimension_conflict

    def test_try_resolve_twice(self, dimension_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not dimension_conflict.is_resolved
        assert isinstance(
            dimension_conflict.try_resolve(),
            dimension_conflict.RemoveDimensionResolution,
        )
        assert dimension_conflict.is_resolved
        assert dimension_conflict.try_resolve() is None

    def test_repr(self, dimension_conflict):
        """Verify the representation of conflict for user interface"""
        assert (
            repr(dimension_conflict) == f"Missing {dimension_conflict.dimension.name}"
        )


class TestAlgorithmConflict:
    """Tests methods related to algorithm configuration conflicts"""

    def test_try_resolve_twice(self, algorithm_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not algorithm_conflict.is_resolved
        assert isinstance(
            algorithm_conflict.try_resolve(), algorithm_conflict.AlgorithmResolution
        )
        assert algorithm_conflict.is_resolved
        assert algorithm_conflict.try_resolve() is None

    def test_try_resolve(self, algorithm_conflict):
        """Verify that resolution is achievable without any input"""
        assert not algorithm_conflict.is_resolved
        resolution = algorithm_conflict.try_resolve()
        assert isinstance(resolution, algorithm_conflict.AlgorithmResolution)
        assert algorithm_conflict.is_resolved
        assert resolution.conflict is algorithm_conflict

    def test_repr(self, old_config, new_config, algorithm_conflict):
        """Verify the representation of conflict for user interface"""
        repr_baseline = "{}\n   !=\n{}".format(
            pprint.pformat(old_config["algorithm"]),
            pprint.pformat(new_config["algorithm"]),
        )
        assert repr(algorithm_conflict) == repr_baseline


class TestOrionVersionConflict:
    """Tests methods related to orion version conflicts"""

    def test_try_resolve_twice(self, orion_version_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not orion_version_conflict.is_resolved
        assert isinstance(
            orion_version_conflict.try_resolve(),
            orion_version_conflict.OrionVersionResolution,
        )
        assert orion_version_conflict.is_resolved
        assert orion_version_conflict.try_resolve() is None

    def test_try_resolve(self, orion_version_conflict):
        """Verify that resolution is achievable without any input"""
        assert not orion_version_conflict.is_resolved
        resolution = orion_version_conflict.try_resolve()
        assert isinstance(resolution, orion_version_conflict.OrionVersionResolution)
        assert orion_version_conflict.is_resolved
        assert resolution.conflict is orion_version_conflict

    def test_repr(self, old_config, new_config, orion_version_conflict):
        """Verify the representation of conflict for user interface"""
        repr_baseline = "{} != {}".format(
            old_config["metadata"]["orion_version"],
            new_config["metadata"]["orion_version"],
        )
        assert repr(orion_version_conflict) == repr_baseline


class TestCodeConflict:
    """Tests methods related to code conflicts"""

    def test_try_resolve_twice(self, code_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not code_conflict.is_resolved
        assert isinstance(
            code_conflict.try_resolve(evc.adapters.CodeChange.BREAK),
            code_conflict.CodeResolution,
        )
        assert code_conflict.is_resolved
        assert code_conflict.try_resolve() is None

    def test_try_resolve(self, code_conflict):
        """Verify that resolution is achievable with valid code change-type input"""
        assert not code_conflict.is_resolved
        resolution = code_conflict.try_resolve(evc.adapters.CodeChange.UNSURE)
        assert isinstance(resolution, code_conflict.CodeResolution)
        assert code_conflict.is_resolved
        assert resolution.conflict is code_conflict
        assert resolution.type == evc.adapters.CodeChange.UNSURE

    def test_try_resolve_bad_code(self, code_conflict):
        """Verify that resolution fails if code is invalid"""
        assert not code_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            code_conflict.try_resolve()
        assert "Invalid code change type 'None'" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            code_conflict.try_resolve("bad-change-type")
        assert "Invalid code change type 'bad-change-type'" in str(exc.value)

    @pytest.mark.usefixtures("mock_infer_versioning_metadata")
    def test_repr(self, code_conflict):
        """Verify the representation of conflict for user interface"""
        assert (
            repr(code_conflict) == "Old hash commit '{'HEAD_sha': 'test', "
            "'active_branch': None, 'diff_sha': 'diff', "
            "'is_dirty': False, 'type': 'git'}'  "
            "!= new hash commit ''to be changed''"
        )

    def test_hash_commit_compar(self):
        """Test that old config hash commit evals to empty."""
        old_config = {"metadata": {"VCS": {}}}
        new_config = {"metadata": {"VCS": {}}}

        assert list(conflict.CodeConflict.detect(old_config, new_config)) == []


class TestCommandLineConflict:
    """Tests methods related to code conflicts"""

    def test_try_resolve_twice(self, cli_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not cli_conflict.is_resolved
        assert isinstance(
            cli_conflict.try_resolve(evc.adapters.CommandLineChange.BREAK),
            cli_conflict.CommandLineResolution,
        )
        assert cli_conflict.is_resolved
        assert cli_conflict.try_resolve() is None

    def test_try_resolve(self, cli_conflict):
        """Verify that resolution is achievable with valid cli change-type input"""
        assert not cli_conflict.is_resolved
        resolution = cli_conflict.try_resolve(evc.adapters.CommandLineChange.UNSURE)
        assert isinstance(resolution, cli_conflict.CommandLineResolution)
        assert cli_conflict.is_resolved
        assert resolution.conflict is cli_conflict
        assert resolution.type == evc.adapters.CommandLineChange.UNSURE

    def test_try_resolve_bad_cli(self, cli_conflict):
        """Verify that resolution fails if cli is invalid"""
        assert not cli_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            cli_conflict.try_resolve()
        assert "Invalid cli change type 'None'" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            cli_conflict.try_resolve("bad-change-type")
        assert "Invalid cli change type 'bad-change-type'" in str(exc.value)

    def test_repr(self, cli_conflict, script_path):
        """Verify the representation of conflict for user interface"""
        assert repr(cli_conflict) == (
            f"Old arguments '_pos_0 {script_path}' != "
            f"new arguments '_pos_0 {script_path} bool-arg True some-new args'"
        )


class TestScriptConfigConflict:
    """Tests methods related to code conflicts"""

    def test_try_resolve_twice(self, config_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not config_conflict.is_resolved
        assert isinstance(
            config_conflict.try_resolve(evc.adapters.ScriptConfigChange.BREAK),
            config_conflict.ScriptConfigResolution,
        )
        assert config_conflict.is_resolved
        assert config_conflict.try_resolve() is None

    def test_try_resolve(self, config_conflict):
        """Verify that resolution is achievable with valid config change-type input"""
        assert not config_conflict.is_resolved
        resolution = config_conflict.try_resolve(evc.adapters.ScriptConfigChange.UNSURE)
        assert isinstance(resolution, config_conflict.ScriptConfigResolution)
        assert config_conflict.is_resolved
        assert resolution.conflict is config_conflict
        assert resolution.type == evc.adapters.ScriptConfigChange.UNSURE

    def test_try_resolve_bad_config(self, config_conflict):
        """Verify that resolution fails if config is invalid"""
        assert not config_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            config_conflict.try_resolve()
        assert "Invalid script's config change type 'None'" in str(exc.value)

        with pytest.raises(ValueError) as exc:
            config_conflict.try_resolve("bad-change-type")
        assert "Invalid script's config change type 'bad-change-type'" in str(exc.value)

    def test_repr(self, config_conflict):
        """Verify the representation of conflict for user interface"""
        assert repr(config_conflict) == "Script's configuration file changed"

    def test_comparison(self, yaml_config, yaml_diff_config, script_path):
        """Test that different configs are detected as conflict."""
        old_config = {
            "metadata": {"user_args": yaml_config, "user_script": script_path}
        }
        new_config = {
            "metadata": {"user_args": yaml_diff_config, "user_script": script_path}
        }

        backward.populate_space(old_config)
        backward.populate_space(new_config)

        conflicts = list(conflict.ScriptConfigConflict.detect(old_config, new_config))
        assert len(conflicts) == 1

    def test_comparison_idem(self, yaml_config, script_path):
        """Test that identical configs are not detected as conflict."""
        old_config = {
            "metadata": {"user_args": yaml_config, "user_script": script_path}
        }
        new_config = {
            "metadata": {
                "user_args": yaml_config + ["--other", "args"],
                "user_script": script_path,
            }
        }

        backward.populate_space(old_config)
        backward.populate_space(new_config)

        assert list(conflict.ScriptConfigConflict.detect(old_config, new_config)) == []


@pytest.mark.usefixtures("orionstate")
class TestExperimentNameConflict:
    """Tests methods related to experiment name conflicts"""

    def test_try_resolve_twice(self, experiment_name_conflict, storage):
        """Verify that conflict cannot be resolved twice"""
        assert not experiment_name_conflict.is_resolved
        assert isinstance(
            experiment_name_conflict.try_resolve("dummy", storage=storage),
            experiment_name_conflict.ExperimentNameResolution,
        )
        assert experiment_name_conflict.is_resolved
        assert experiment_name_conflict.try_resolve(storage=storage) is None

    def test_try_resolve(self, experiment_name_conflict, storage):
        """Verify that resolution is achievable with a valid name"""
        new_name = "dummy"
        assert not experiment_name_conflict.is_resolved
        resolution = experiment_name_conflict.try_resolve(new_name, storage=storage)
        assert isinstance(resolution, experiment_name_conflict.ExperimentNameResolution)
        assert experiment_name_conflict.is_resolved
        assert resolution.conflict is experiment_name_conflict
        assert resolution.new_name == new_name

    def test_branch_w_existing_exp(self, existing_exp_conflict, storage):
        """Test branching when an existing experiment with the new name already exists"""
        with pytest.raises(ValueError) as exc:
            existing_exp_conflict.try_resolve("dummy", storage=storage)

        assert "Cannot" in str(exc.value)

    def test_conflict_exp_no_child(self, exp_no_child_conflict, storage):
        """Verify the version number is incremented when exp has no child."""
        new_name = "test"
        assert not exp_no_child_conflict.is_resolved
        resolution = exp_no_child_conflict.try_resolve(new_name, storage=storage)
        assert isinstance(resolution, exp_no_child_conflict.ExperimentNameResolution)
        assert exp_no_child_conflict.is_resolved
        assert resolution.conflict is exp_no_child_conflict
        assert resolution.old_version == 1
        assert resolution.new_version == 2

    def test_conflict_exp_w_child(self, exp_w_child_conflict, storage):
        """Verify the version number is incremented from child when exp has a child."""
        new_name = "test"
        assert not exp_w_child_conflict.is_resolved
        resolution = exp_w_child_conflict.try_resolve(new_name, storage=storage)
        assert isinstance(resolution, exp_w_child_conflict.ExperimentNameResolution)
        assert exp_w_child_conflict.is_resolved
        assert resolution.conflict is exp_w_child_conflict
        assert resolution.new_version == 3

    def test_conflict_exp_w_child_as_parent(
        self, exp_w_child_as_parent_conflict, storage
    ):
        """Verify that an error is raised when trying to branch from parent."""
        new_name = "test"
        with pytest.raises(ValueError) as exc:
            exp_w_child_as_parent_conflict.try_resolve(new_name, storage=storage)

        assert "Experiment name" in str(exc.value)

    def test_conflict_exp_renamed(self, exp_w_child_conflict, storage):
        """Verify the version number is not incremented when exp is renamed."""
        # It increments from child
        new_name = "test2"
        assert not exp_w_child_conflict.is_resolved
        resolution = exp_w_child_conflict.try_resolve(new_name, storage=storage)
        assert isinstance(resolution, exp_w_child_conflict.ExperimentNameResolution)
        assert exp_w_child_conflict.is_resolved
        assert resolution.conflict is exp_w_child_conflict
        assert resolution.old_version == 2
        assert resolution.new_version == 1

    def test_repr(self, experiment_name_conflict):
        """Verify the representation of conflict for user interface"""
        assert (
            repr(experiment_name_conflict)
            == "Experiment name 'test' already exist with version '1'"
        )


class TestConflicts:
    """Tests for Conflicts container, for getter, deprecation and resolve methods"""

    def test_register(self, conflicts):
        """Verify that conflicts are properly registered"""
        assert len(conflicts.conflicts) == 7
        isinstance(conflicts.conflicts[0], evc.conflicts.NewDimensionConflict)
        isinstance(conflicts.conflicts[-1], evc.conflicts.ExperimentNameConflict)

    def test_get(self, conflicts):
        """Verify that bare get() fetches all conflicts"""
        assert len(conflicts.get()) == 7
        isinstance(conflicts.get()[0], evc.conflicts.NewDimensionConflict)
        isinstance(conflicts.get()[-1], evc.conflicts.ExperimentNameConflict)

    def test_get_one_type(self, conflicts):
        """Verify that get() fetches proper given type"""
        found_conflicts = conflicts.get([evc.conflicts.NewDimensionConflict])
        assert len(found_conflicts) == 1
        assert isinstance(found_conflicts[0], evc.conflicts.NewDimensionConflict)

    def test_get_many_types(self, conflicts):
        """Verify that get() fetches many types properly"""
        types = (
            evc.conflicts.NewDimensionConflict,
            evc.conflicts.AlgorithmConflict,
            evc.conflicts.ExperimentNameConflict,
        )
        found_conflicts = conflicts.get(types)
        assert len(found_conflicts) == 3
        assert isinstance(found_conflicts[0], types)
        assert isinstance(found_conflicts[1], types)
        assert isinstance(found_conflicts[2], types)

    def test_get_dimension_name(self, conflicts):
        """Verify that get() fetch the conflict associated to given dimension_name"""
        found_conflicts = conflicts.get(dimension_name="missing")
        assert len(found_conflicts) == 1
        assert isinstance(found_conflicts[0], evc.conflicts.MissingDimensionConflict)
        assert found_conflicts[0].dimension.name == "missing"

    def test_get_invalid_dimension_name(self, conflicts):
        """Verify that get() raises when dimension_name is invalid"""
        with pytest.raises(ValueError) as exc:
            conflicts.get(dimension_name="invalid")
        assert "Dimension name 'invalid' not found in conflicts" in str(exc.value)

    def test_get_callback(self, conflicts):
        """Verify that get() callback are effectively used"""

        def always_true(conflict):
            return True

        found_conflicts = conflicts.get(callback=always_true)
        assert len(found_conflicts) == 7
        isinstance(found_conflicts[0], evc.conflicts.NewDimensionConflict)
        isinstance(found_conflicts[-1], evc.conflicts.ExperimentNameConflict)

        def always_false(conflict):
            return False

        assert len(conflicts.get(callback=always_false)) == 0

    def test_get_remaining(self, conflicts):
        """Verify that get_remaining() only fetch non resolved conflicts"""
        assert len(conflicts.get_remaining()) == 7
        conflicts.get_remaining()[0].try_resolve()
        assert len(conflicts.get_remaining()) == 6
        conflicts.get_remaining()[0].try_resolve()
        assert len(conflicts.get_remaining()) == 5

    def test_get_resolved(self, conflicts):
        """Verify that get_resolved() only fetch resolved conflicts"""
        assert len(conflicts.get_resolved()) == 0
        conflicts.get_remaining()[0].try_resolve()
        assert len(conflicts.get_resolved()) == 1
        conflicts.get_remaining()[0].try_resolve()
        assert len(conflicts.get_resolved()) == 2

    def test_get_resolutions(
        self, conflicts, new_dimension_conflict, missing_dimension_conflict
    ):
        """Verify that get_resolution() only fetch unique resolutions"""
        assert len(list(conflicts.get_resolutions())) == 0
        conflicts.try_resolve(
            missing_dimension_conflict, new_dimension_conflict=new_dimension_conflict
        )
        assert len(list(conflicts.get_resolutions())) == 1
        assert len(list(conflicts.get_resolved())) == 2
        conflicts.get_remaining()[0].try_resolve()
        assert len(list(conflicts.get_resolutions())) == 2
        assert len(conflicts.get_resolved()) == 3

    def test_are_resolved(
        self, new_dimension_conflict, missing_dimension_conflict, algorithm_conflict
    ):
        """Verify if resolution status is correct"""
        conflicts = evc.conflicts.Conflicts()
        conflicts.register(new_dimension_conflict)
        conflicts.register(missing_dimension_conflict)
        conflicts.register(algorithm_conflict)

        assert len(conflicts.get_remaining()) == 3
        assert not conflicts.are_resolved
        conflicts.get_remaining()[0].try_resolve()

        assert len(conflicts.get_remaining()) == 2
        assert not conflicts.are_resolved
        conflicts.get_remaining()[0].try_resolve()

        assert len(conflicts.get_remaining()) == 1
        assert not conflicts.are_resolved
        conflicts.get_remaining()[0].try_resolve()

        assert len(conflicts.get_remaining()) == 0
        assert conflicts.are_resolved

    def test_deprecate_existing_conflict(self, new_dimension_conflict, conflicts):
        """Verify deprecated conflicts are not inside conflicts anymore"""
        assert len(conflicts.conflicts) == 7
        conflicts.deprecate([new_dimension_conflict])
        assert len(conflicts.conflicts) == 6

    def test_deprecate_non_existing_conflict(self, conflicts):
        """Verify attempt to deprecate non-existing conflicts raises ValueError"""
        assert len(conflicts.conflicts) == 7
        with pytest.raises(ValueError) as exc:
            conflicts.deprecate(["dummy object"])
        assert "'dummy object' is not in list" in str(exc.value)

    def test_try_resolve_silence_errors(self, capsys, code_conflict, conflicts):
        """Verify try_resolve errors are silenced"""
        conflicts.try_resolve(code_conflict)
        out, err = capsys.readouterr()
        assert (
            out.split("\n")[-3]
            == "ValueError: Invalid code change type 'None'. Should be one of "
            "['noeffect', 'break', 'unsure']"
        )

        conflicts.try_resolve(code_conflict, silence_errors=True)
        out, err = capsys.readouterr()
        assert out == ""

    def test_try_resolve_keyboard_interrupt(self, capsys, code_conflict, conflicts):
        """Verify try_resolve catch all errors but not keyboard interrupt"""

        def raise_base_exception(*args, **kwargs):
            raise Exception("Raised properly")

        code_conflict.try_resolve = raise_base_exception
        conflicts.try_resolve(code_conflict)
        out, err = capsys.readouterr()
        assert out.split("\n")[-3] == "Exception: Raised properly"

        def raise_keyboard_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()

        code_conflict.try_resolve = raise_keyboard_interrupt
        with pytest.raises(KeyboardInterrupt):
            conflicts.try_resolve(code_conflict)

    def test_try_resolve_side_effect_conflicts(
        self, new_dimension_conflict, missing_dimension_conflict, conflicts
    ):
        """Verify try_resolve register new conflicts created by resolutions"""
        assert len(conflicts.conflicts) == 7
        conflicts.try_resolve(
            missing_dimension_conflict, new_dimension_conflict=new_dimension_conflict
        )
        assert len(conflicts.conflicts) == 8
        assert isinstance(
            conflicts.conflicts[-1], evc.conflicts.ChangedDimensionConflict
        )
        assert not conflicts.conflicts[-1].is_resolved
