#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.evc.conflicts`."""

import pprint

import pytest

from orion.algo.space import Dimension
from orion.core import evc


@pytest.fixture
def conflicts(new_dimension_conflict, changed_dimension_conflict, missing_dimension_conflict,
              algorithm_conflict, code_conflict, experiment_name_conflict):
    """Create a container for conflicts with one of each types for testing purposes"""
    conflicts = evc.conflicts.Conflicts()
    conflicts.register(new_dimension_conflict)
    conflicts.register(changed_dimension_conflict)
    conflicts.register(missing_dimension_conflict)
    conflicts.register(algorithm_conflict)
    conflicts.register(code_conflict)
    conflicts.register(experiment_name_conflict)
    return conflicts


class TestNewDimensionConflict(object):
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
        assert isinstance(resolution, new_dimension_with_default_conflict.AddDimensionResolution)
        assert (resolution.default_value ==
                new_dimension_with_default_conflict.dimension.default_value)

    def test_try_resolve_default(self, new_dimension_conflict):
        """Verify that resolution includes default value if provided"""
        default_value = 1.2
        assert not new_dimension_conflict.is_resolved
        resolution = new_dimension_conflict.try_resolve(default_value)
        assert new_dimension_conflict.is_resolved
        assert isinstance(resolution, new_dimension_conflict.AddDimensionResolution)
        assert (resolution.default_value == default_value)

    def test_try_resolve_default_over_dim(self, new_dimension_with_default_conflict):
        """Verify that resolution overwrites dimension's default value if one is provided"""
        default_value = 1.2
        assert not new_dimension_with_default_conflict.is_resolved
        resolution = new_dimension_with_default_conflict.try_resolve(default_value)
        assert new_dimension_with_default_conflict.is_resolved
        assert isinstance(resolution, new_dimension_with_default_conflict.AddDimensionResolution)
        assert (resolution.default_value == default_value)

    def test_try_resolve_bad_default(self, new_dimension_conflict):
        """Verify that resolution fails if default value is invalid"""
        assert not new_dimension_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            new_dimension_conflict.try_resolve('bad-default')
        assert "could not convert string to float: 'bad-default'" in str(exc.value)

    def test_try_resolve_twice(self, new_dimension_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not new_dimension_conflict.is_resolved
        assert isinstance(new_dimension_conflict.try_resolve(),
                          new_dimension_conflict.AddDimensionResolution)
        assert new_dimension_conflict.is_resolved
        assert new_dimension_conflict.try_resolve() is None

    def test_repr(self, new_dimension_conflict):
        """Verify the representation of conflict for user interface"""
        assert repr(new_dimension_conflict) == "New new"


class TestChangedDimensionConflict(object):
    """Tests methods related to changed dimension prior conflicts"""

    def test_try_resolve_twice(self, changed_dimension_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not changed_dimension_conflict.is_resolved
        assert isinstance(changed_dimension_conflict.try_resolve(),
                          changed_dimension_conflict.ChangeDimensionResolution)
        assert changed_dimension_conflict.is_resolved
        assert changed_dimension_conflict.try_resolve() is None

    def test_try_resolve(self, changed_dimension_conflict):
        """Verify that resolution is achievable without any input"""
        assert not changed_dimension_conflict.is_resolved
        resolution = changed_dimension_conflict.try_resolve()
        assert isinstance(resolution, changed_dimension_conflict.ChangeDimensionResolution)
        assert changed_dimension_conflict.is_resolved
        assert resolution.conflict is changed_dimension_conflict

    def test_repr(self, changed_dimension_conflict):
        """Verify the representation of conflict for user interface"""
        assert (repr(changed_dimension_conflict) ==
                "changed~uniform(-10, 10) != changed~normal(0, 2)")


class TestMissingDimensionConflict(object):
    """Tests methods related to missing dimension conflicts"""

    def test_try_resolve_no_default(self, missing_dimension_conflict):
        """Verify that resolution is achievable without default value provided"""
        assert not missing_dimension_conflict.is_resolved
        resolution = missing_dimension_conflict.try_resolve()
        assert missing_dimension_conflict.is_resolved
        assert isinstance(resolution, missing_dimension_conflict.RemoveDimensionResolution)
        assert resolution.default_value is Dimension.NO_DEFAULT_VALUE

    def test_try_resolve_default_from_dim(self, missing_dimension_with_default_conflict):
        """Verify that resolution uses default value from dim when no provided by user"""
        assert not missing_dimension_with_default_conflict.is_resolved
        resolution = missing_dimension_with_default_conflict.try_resolve()
        assert missing_dimension_with_default_conflict.is_resolved
        assert isinstance(resolution,
                          missing_dimension_with_default_conflict.RemoveDimensionResolution)
        assert (resolution.default_value ==
                missing_dimension_with_default_conflict.dimension.default_value)

    def test_try_resolve_default(self, missing_dimension_conflict):
        """Verify that resolution uses default value provided by user"""
        default_value = 1.2
        assert not missing_dimension_conflict.is_resolved
        resolution = missing_dimension_conflict.try_resolve(default_value=default_value)
        assert missing_dimension_conflict.is_resolved
        assert isinstance(resolution, missing_dimension_conflict.RemoveDimensionResolution)
        assert (resolution.default_value == default_value)

    def test_try_resolve_default_over_dim(self, missing_dimension_with_default_conflict):
        """Verify that resolution overwrite dimension's default value when user provide one"""
        default_value = 1.2
        assert not missing_dimension_with_default_conflict.is_resolved
        resolution = missing_dimension_with_default_conflict.try_resolve(
            default_value=default_value)
        assert missing_dimension_with_default_conflict.is_resolved
        assert isinstance(resolution,
                          missing_dimension_with_default_conflict.RemoveDimensionResolution)
        assert (resolution.default_value == default_value)

    def test_try_resolve_bad_default(self, missing_dimension_conflict):
        """Verify that resolution fails if default value is invalid"""
        assert not missing_dimension_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            missing_dimension_conflict.try_resolve(default_value='-100')
        assert "Default value `-100.0` is outside" in str(exc.value)

    def test_try_resolve_renaming(self, missing_dimension_conflict, new_dimension_conflict):
        """Verify that resolution is a renaming when new_dimension_conflict is provided"""
        assert not missing_dimension_conflict.is_resolved
        resolution = missing_dimension_conflict.try_resolve(new_dimension_conflict)
        assert isinstance(resolution, missing_dimension_conflict.RenameDimensionResolution)
        assert resolution.conflict is missing_dimension_conflict
        assert resolution.new_dimension_conflict is new_dimension_conflict

    def test_try_resolve_twice(self, missing_dimension_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not missing_dimension_conflict.is_resolved
        assert isinstance(missing_dimension_conflict.try_resolve(),
                          missing_dimension_conflict.RemoveDimensionResolution)
        assert missing_dimension_conflict.is_resolved
        assert missing_dimension_conflict.try_resolve() is None

    def test_repr(self, missing_dimension_conflict):
        """Verify the representation of conflict for user interface"""
        assert repr(missing_dimension_conflict) == "Missing missing"


class TestAlgorithmConflict(object):
    """Tests methods related to algorithm configuration conflicts"""

    def test_try_resolve_twice(self, algorithm_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not algorithm_conflict.is_resolved
        assert isinstance(algorithm_conflict.try_resolve(),
                          algorithm_conflict.AlgorithmResolution)
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
        repr_baseline = "{0}\n   !=\n{1}".format(pprint.pformat(old_config['algorithms']),
                                                 pprint.pformat(new_config['algorithms']))
        assert repr(algorithm_conflict) == repr_baseline


class TestCodeConflict(object):
    """Tests methods related to code conflicts"""

    def test_try_resolve_twice(self, code_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not code_conflict.is_resolved
        assert isinstance(code_conflict.try_resolve(evc.adapters.CodeChange.BREAK),
                          code_conflict.CodeResolution)
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
        assert repr(code_conflict) == "Old hash commit '{'type': 'git', "\
                                      "'is_dirty': False, 'HEAD_sha': 'test',"\
                                      " 'active_branch': None, 'diff_sha': 'diff'}'"\
                                      " != new hash commit 'to be changed'"


class TestCommandLineConflict(object):
    """Tests methods related to code conflicts"""

    def test_try_resolve_twice(self, cli_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not cli_conflict.is_resolved
        assert isinstance(cli_conflict.try_resolve(evc.adapters.CommandLineChange.BREAK),
                          cli_conflict.CommandLineResolution)
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

    def test_repr(self, cli_conflict):
        """Verify the representation of conflict for user interface"""
        assert repr(cli_conflict) == "Old arguments '' != new arguments '--some-new=args'"


class TestScriptConfigConflict(object):
    """Tests methods related to code conflicts"""

    def test_try_resolve_twice(self, config_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not config_conflict.is_resolved
        assert isinstance(config_conflict.try_resolve(evc.adapters.ScriptConfigChange.BREAK),
                          config_conflict.ScriptConfigResolution)
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


@pytest.mark.usefixtures("create_db_instance")
class TestExperimentNameConflict(object):
    """Tests methods related to experiment name conflicts"""

    def test_try_resolve_twice(self, experiment_name_conflict):
        """Verify that conflict cannot be resolved twice"""
        assert not experiment_name_conflict.is_resolved
        assert isinstance(experiment_name_conflict.try_resolve("dummy"),
                          experiment_name_conflict.ExperimentNameResolution)
        assert experiment_name_conflict.is_resolved
        assert experiment_name_conflict.try_resolve() is None

    def test_try_resolve_bad_name(self, experiment_name_conflict):
        """Verify that resolution fails if name already exist in db"""
        assert not experiment_name_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            experiment_name_conflict.try_resolve("test")
        assert "Experiment name 'test' already exist" in str(exc.value)

        assert not experiment_name_conflict.is_resolved
        with pytest.raises(ValueError) as exc:
            experiment_name_conflict.try_resolve()
        assert "No new name provided." in str(exc.value)

    def test_try_resolve(self, experiment_name_conflict):
        """Verify that resolution is achievable with a valid name"""
        new_name = "dummy"
        assert not experiment_name_conflict.is_resolved
        resolution = experiment_name_conflict.try_resolve(new_name)
        assert isinstance(resolution, experiment_name_conflict.ExperimentNameResolution)
        assert experiment_name_conflict.is_resolved
        assert resolution.conflict is experiment_name_conflict
        assert resolution.new_name == new_name

    def test_repr(self, experiment_name_conflict):
        """Verify the representation of conflict for user interface"""
        assert (repr(experiment_name_conflict) ==
                "Experiment name 'test' already exist for user 'some_user_name'")


class TestConflicts(object):
    """Tests for Conflicts container, for getter, deprecation and resolve methods"""

    def test_register(self, conflicts):
        """Verify that conflicts are properly registerd"""
        assert len(conflicts.conflicts) == 6
        isinstance(conflicts.conflicts[0], evc.conflicts.NewDimensionConflict)
        isinstance(conflicts.conflicts[-1], evc.conflicts.ExperimentNameConflict)

    def test_get(self, conflicts):
        """Verify that bare get() fetches all conflicts"""
        assert len(conflicts.get()) == 6
        isinstance(conflicts.get()[0], evc.conflicts.NewDimensionConflict)
        isinstance(conflicts.get()[-1], evc.conflicts.ExperimentNameConflict)

    def test_get_one_type(self, conflicts):
        """Verify that get() fetches proper given type"""
        found_conflicts = conflicts.get([evc.conflicts.NewDimensionConflict])
        assert len(found_conflicts) == 1
        assert isinstance(found_conflicts[0], evc.conflicts.NewDimensionConflict)

    def test_get_many_types(self, conflicts):
        """Verify that get() fetches many types properly"""
        types = (evc.conflicts.NewDimensionConflict, evc.conflicts.AlgorithmConflict,
                 evc.conflicts.ExperimentNameConflict)
        found_conflicts = conflicts.get(types)
        assert len(found_conflicts) == 3
        assert isinstance(found_conflicts[0], types)
        assert isinstance(found_conflicts[1], types)
        assert isinstance(found_conflicts[2], types)

    def test_get_dimension_name(self, conflicts):
        """Verify that get() fetch the conflict associated to given dimension_name"""
        found_conflicts = conflicts.get(dimension_name='missing')
        assert len(found_conflicts) == 1
        assert isinstance(found_conflicts[0], evc.conflicts.MissingDimensionConflict)
        assert found_conflicts[0].dimension.name == 'missing'

    def test_get_invalid_dimension_name(self, conflicts):
        """Verify that get() raises when dimension_name is invalid"""
        with pytest.raises(ValueError) as exc:
            conflicts.get(dimension_name='invalid')
        assert "Dimension name 'invalid' not found in conflicts" in str(exc.value)

    def test_get_callback(self, conflicts):
        """Verify that get() callback are effectively used"""
        def always_true(conflict):
            return True

        found_conflicts = conflicts.get(callback=always_true)
        assert len(found_conflicts) == 6
        isinstance(found_conflicts[0], evc.conflicts.NewDimensionConflict)
        isinstance(found_conflicts[-1], evc.conflicts.ExperimentNameConflict)

        def always_false(conflict):
            return False

        assert len(conflicts.get(callback=always_false)) == 0

    def test_get_remaining(self, conflicts):
        """Verify that get_remaining() only fetch non resolved conflicts"""
        assert len(conflicts.get_remaining()) == 6
        conflicts.get_remaining()[0].try_resolve()
        assert len(conflicts.get_remaining()) == 5
        conflicts.get_remaining()[0].try_resolve()
        assert len(conflicts.get_remaining()) == 4

    def test_get_resolved(self, conflicts):
        """Verify that get_resolved() only fetch resolved conflicts"""
        assert len(conflicts.get_resolved()) == 0
        conflicts.get_remaining()[0].try_resolve()
        assert len(conflicts.get_resolved()) == 1
        conflicts.get_remaining()[0].try_resolve()
        assert len(conflicts.get_resolved()) == 2

    def test_get_resolutions(self, conflicts, new_dimension_conflict, missing_dimension_conflict):
        """Verify that get_resolution() only fetch unique resolutions"""
        assert len(list(conflicts.get_resolutions())) == 0
        conflicts.try_resolve(missing_dimension_conflict,
                              new_dimension_conflict=new_dimension_conflict)
        assert len(list(conflicts.get_resolutions())) == 1
        assert len(list(conflicts.get_resolved())) == 2
        conflicts.get_remaining()[0].try_resolve()
        assert len(list(conflicts.get_resolutions())) == 2
        assert len(conflicts.get_resolved()) == 3

    def test_are_resolved(self, new_dimension_conflict, missing_dimension_conflict,
                          algorithm_conflict):
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
        assert len(conflicts.conflicts) == 6
        conflicts.deprecate([new_dimension_conflict])
        assert len(conflicts.conflicts) == 5

    def test_deprecate_non_existing_conflict(self, conflicts):
        """Verify attempt to deprecate non-existing conflicts raises ValueError"""
        assert len(conflicts.conflicts) == 6
        with pytest.raises(ValueError) as exc:
            conflicts.deprecate(["dummy object"])
        assert "'dummy object' is not in list" in str(exc.value)

    def test_try_resolve_silence_errors(self, capsys, experiment_name_conflict, conflicts):
        """Verify try_resolve errors are silenced"""
        conflicts.try_resolve(experiment_name_conflict)
        out, err = capsys.readouterr()
        assert (out.split("\n")[-3] ==
                "ValueError: No new name provided. Cannot resolve experiment name conflict.")

        conflicts.try_resolve(experiment_name_conflict, silence_errors=True)
        out, err = capsys.readouterr()
        assert out == ''

    def test_try_resolve_keyboard_interrupt(self, capsys, code_conflict, conflicts):
        """Verify try_resolve catch all errors but not keyboard interrupt"""
        def raise_base_exception(*args, **kwargs):
            raise Exception("Raised properly")
        code_conflict.try_resolve = raise_base_exception
        conflicts.try_resolve(code_conflict)
        out, err = capsys.readouterr()
        assert (out.split("\n")[-3] == "Exception: Raised properly")

        def raise_keyboard_interrupt(*args, **kwargs):
            raise KeyboardInterrupt()
        code_conflict.try_resolve = raise_keyboard_interrupt
        with pytest.raises(KeyboardInterrupt):
            conflicts.try_resolve(code_conflict)

    def test_try_resolve_side_effect_conflicts(
            self, new_dimension_conflict,
            missing_dimension_conflict, conflicts):
        """Verify try_resolve register new conflicts created by resolutions"""
        assert len(conflicts.conflicts) == 6
        conflicts.try_resolve(missing_dimension_conflict,
                              new_dimension_conflict=new_dimension_conflict)
        assert len(conflicts.conflicts) == 7
        assert isinstance(conflicts.conflicts[-1], evc.conflicts.ChangedDimensionConflict)
        assert not conflicts.conflicts[-1].is_resolved
