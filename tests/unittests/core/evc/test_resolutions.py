#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests resolutions in :mod:`orion.core.evc.conflicts`."""

import pytest

from orion.algo.space import Dimension
from orion.core.evc import adapters, conflicts


@pytest.fixture
def add_dimension_resolution(new_dimension_conflict):
    """Create a resolution for a new dimension"""
    return new_dimension_conflict.AddDimensionResolution(new_dimension_conflict)


@pytest.fixture
def change_dimension_resolution(changed_dimension_conflict):
    """Create a resolution for a changed prior"""
    return changed_dimension_conflict.ChangeDimensionResolution(
        changed_dimension_conflict
    )


@pytest.fixture
def remove_dimension_resolution(missing_dimension_conflict):
    """Create a resolution to remove a missing dimension"""
    return missing_dimension_conflict.RemoveDimensionResolution(
        missing_dimension_conflict, default_value=0
    )


@pytest.fixture
def rename_dimension_resolution(missing_dimension_conflict, new_dimension_conflict):
    """Create a resolution to rename a missing dimension to a new dimension"""
    return missing_dimension_conflict.RenameDimensionResolution(
        missing_dimension_conflict, new_dimension_conflict
    )


@pytest.fixture
def algorithm_resolution(algorithm_conflict):
    """Create a resolution for a new algorithm configuration"""
    return algorithm_conflict.AlgorithmResolution(algorithm_conflict)


@pytest.fixture
def orion_version_resolution(orion_version_conflict):
    """Create a resolution for a new orion version"""
    return orion_version_conflict.OrionVersionResolution(orion_version_conflict)


@pytest.fixture
def code_resolution(code_conflict):
    """Create a resolution for a code conflict"""
    return code_conflict.CodeResolution(code_conflict, adapters.CodeChange.BREAK)


@pytest.fixture
def experiment_name_resolution(setup_pickleddb_database, experiment_name_conflict):
    """Create a resolution for a code conflict"""
    return experiment_name_conflict.ExperimentNameResolution(
        experiment_name_conflict, new_name="new-exp-name"
    )


class TestAddDimensionResolution(object):
    """Test methods for resolution of new dimension"""

    def test_init_no_default(self, new_dimension_conflict):
        """Verify instantiation of resolution without default value"""
        resolution = new_dimension_conflict.AddDimensionResolution(
            new_dimension_conflict
        )
        assert resolution.default_value is Dimension.NO_DEFAULT_VALUE

    def test_init_no_default_but_in_dim(self, new_dimension_with_default_conflict):
        """Verify instantiation of resolution with default value in dimension"""
        resolution = new_dimension_with_default_conflict.AddDimensionResolution(
            new_dimension_with_default_conflict
        )
        assert (
            resolution.default_value
            == new_dimension_with_default_conflict.dimension.default_value
        )

    def test_init_with_default(
        self, new_dimension_conflict, new_dimension_with_default_conflict
    ):
        """Verify instantiation of resolution with default value given by user"""
        default_value = 1.1
        resolution = new_dimension_conflict.AddDimensionResolution(
            new_dimension_conflict, default_value=default_value
        )
        assert resolution.default_value == default_value

        assert (
            new_dimension_with_default_conflict.dimension.default_value != default_value
        )
        resolution = new_dimension_conflict.AddDimensionResolution(
            new_dimension_with_default_conflict, default_value=default_value
        )
        assert resolution.default_value == default_value

    def test_init_bad_default(self, new_dimension_conflict):
        """Verify instantiation of resolution with invalid default value"""
        default_value = "bad"
        with pytest.raises(ValueError) as exc:
            new_dimension_conflict.AddDimensionResolution(
                new_dimension_conflict, default_value=default_value
            )
        assert "could not convert string to float: 'bad'" in str(exc.value)

    def test_new_prior_no_default(self, new_dimension_conflict):
        """Verify prior string without default value"""
        resolution = new_dimension_conflict.AddDimensionResolution(
            new_dimension_conflict
        )
        assert resolution.new_prior == "norm(0, 2)"

    def test_new_prior_default_from_dim(self, new_dimension_with_default_conflict):
        """Verify prior string with default value in dimension"""
        resolution = new_dimension_with_default_conflict.AddDimensionResolution(
            new_dimension_with_default_conflict
        )
        assert resolution.new_prior == "norm(0, 2, default_value=0.001)"

    def test_new_prior_default(
        self, new_dimension_conflict, new_dimension_with_default_conflict
    ):
        """Verify prior string with new default value given by user"""
        default_value = 1.2
        resolution = new_dimension_with_default_conflict.AddDimensionResolution(
            new_dimension_with_default_conflict, default_value=default_value
        )
        assert resolution.new_prior == "norm(0, 2, default_value={})".format(
            default_value
        )

        resolution = new_dimension_conflict.AddDimensionResolution(
            new_dimension_conflict, default_value=default_value
        )
        assert resolution.new_prior == "norm(0, 2, default_value={})".format(
            default_value
        )

    def test_prefix(self, add_dimension_resolution):
        """Verify prefix of resolution with corresponding marker"""
        assert add_dimension_resolution.prefix == "new~+"

    def test_repr_without_default(self, add_dimension_resolution):
        """Verify resolution representation for user interface, without default value"""
        assert repr(add_dimension_resolution) == "new~+norm(0, 2)"

    def test_repr_default_from_dim(self, new_dimension_with_default_conflict):
        """Verify resolution representation for user interface, without default value"""
        resolution = new_dimension_with_default_conflict.AddDimensionResolution(
            new_dimension_with_default_conflict
        )
        assert repr(resolution) == "new~+norm(0, 2, default_value=0.001)"

    def test_adapters_without_default(self, new_dimension_conflict):
        """Verify adapters without default values (filter everything out)"""
        param = {"name": "new", "type": "real", "value": Dimension.NO_DEFAULT_VALUE}
        resolution = new_dimension_conflict.AddDimensionResolution(
            new_dimension_conflict
        )
        resolution_adapters = resolution.get_adapters()
        assert len(resolution_adapters) == 1
        assert (
            resolution_adapters[0].configuration
            == adapters.DimensionAddition(param).configuration
        )

    def test_adapters_with_default(self, new_dimension_conflict):
        """Verify adapters with default values"""
        param = {"name": "new", "type": "real", "value": 1.1}
        resolution = new_dimension_conflict.AddDimensionResolution(
            new_dimension_conflict, default_value=1.1
        )
        resolution_adapters = resolution.get_adapters()
        assert len(resolution_adapters) == 1
        assert (
            resolution_adapters[0].configuration
            == adapters.DimensionAddition(param).configuration
        )

    def test_revert(self, new_dimension_conflict, add_dimension_resolution):
        """Verify reverting resolution set conflict to unresolved"""
        assert new_dimension_conflict.is_resolved
        assert add_dimension_resolution.revert() == []
        assert not new_dimension_conflict.is_resolved
        assert new_dimension_conflict.resolution is None


class TestChangeDimensionResolution(object):
    """Test methods for resolution of changed dimensions"""

    def test_prefix(self, change_dimension_resolution):
        """Verify prefix of resolution with corresponding marker"""
        assert change_dimension_resolution.prefix == "changed~+"

    def test_repr(self, change_dimension_resolution):
        """Verify resolution representation for user interface"""
        assert repr(change_dimension_resolution) == "changed~+normal(0, 2)"

    def test_adapters(self, change_dimension_resolution):
        """Verify adapters with old and new priors"""
        name = "changed"
        old_prior = "uniform(-10, 10)"
        new_prior = "normal(0, 2)"
        resolution_adapters = change_dimension_resolution.get_adapters()
        assert len(resolution_adapters) == 1
        assert (
            resolution_adapters[0].configuration
            == adapters.DimensionPriorChange(name, old_prior, new_prior).configuration
        )

    def test_revert(self, changed_dimension_conflict, change_dimension_resolution):
        """Verify reverting resolution set conflict to unresolved"""
        assert changed_dimension_conflict.is_resolved
        assert change_dimension_resolution.revert() == []
        assert not changed_dimension_conflict.is_resolved
        assert changed_dimension_conflict.resolution is None


class TestRemoveDimensionResolution(object):
    """Test methods for resolution of missing dimensions"""

    def test_prefix(self, missing_dimension_conflict):
        """Verify prefix of resolution with corresponding marker"""
        resolution = missing_dimension_conflict.RemoveDimensionResolution(
            missing_dimension_conflict
        )
        assert resolution.prefix == "missing~-"

    def test_repr_no_default(self, missing_dimension_conflict):
        """Verify resolution representation for user interface, without default value"""
        resolution = missing_dimension_conflict.RemoveDimensionResolution(
            missing_dimension_conflict
        )
        assert repr(resolution) == "missing~-"

    def test_repr_default_from_dim(self, missing_dimension_with_default_conflict):
        """Verify resolution representation for user interface, with default value from dimension"""
        resolution = missing_dimension_with_default_conflict.RemoveDimensionResolution(
            missing_dimension_with_default_conflict
        )
        assert repr(resolution) == "missing~-0.0"

    def test_repr_default(
        self, missing_dimension_conflict, missing_dimension_with_default_conflict
    ):
        """Verify resolution representation for user interface, with default provided by user"""
        default_value = 1.2
        resolution = missing_dimension_conflict.RemoveDimensionResolution(
            missing_dimension_with_default_conflict, default_value=default_value
        )
        assert repr(resolution) == "missing~-{}".format(default_value)

        resolution = missing_dimension_conflict.RemoveDimensionResolution(
            missing_dimension_conflict, default_value=default_value
        )
        assert repr(resolution) == "missing~-{}".format(default_value)

    def test_adapters_without_default(self, missing_dimension_conflict):
        """Verify adapters without default value"""
        param = {"name": "missing", "type": "real", "value": Dimension.NO_DEFAULT_VALUE}
        resolution = missing_dimension_conflict.RemoveDimensionResolution(
            missing_dimension_conflict
        )
        resolution_adapters = resolution.get_adapters()
        assert len(resolution_adapters) == 1
        assert (
            resolution_adapters[0].configuration
            == adapters.DimensionDeletion(param).configuration
        )

    def test_adapters_with_default(self, missing_dimension_conflict):
        """Verify adapters with default value"""
        param = {"name": "missing", "type": "real", "value": 1.2}
        resolution = missing_dimension_conflict.RemoveDimensionResolution(
            missing_dimension_conflict, default_value=1.2
        )
        resolution_adapters = resolution.get_adapters()
        assert len(resolution_adapters) == 1
        assert (
            resolution_adapters[0].configuration
            == adapters.DimensionDeletion(param).configuration
        )

    def test_revert(self, missing_dimension_conflict, remove_dimension_resolution):
        """Verify reverting resolution set conflict to unresolved"""
        assert missing_dimension_conflict.is_resolved
        assert remove_dimension_resolution.revert() == []
        assert not missing_dimension_conflict.is_resolved
        assert missing_dimension_conflict.resolution is None


class TestRenameDimensionResolution(object):
    """Test methods for renaming of missing dimensions"""

    def test_init_same_prior(
        self, missing_dimension_conflict, new_dimension_same_prior_conflict
    ):
        """Verify initialisation with identical priors generates no side-effect conflicts"""
        assert not missing_dimension_conflict.is_resolved
        assert not new_dimension_same_prior_conflict.is_resolved
        resolution = missing_dimension_conflict.RenameDimensionResolution(
            missing_dimension_conflict, new_dimension_same_prior_conflict
        )
        assert missing_dimension_conflict.is_resolved
        assert new_dimension_same_prior_conflict.is_resolved
        assert resolution.new_conflicts == []

    def test_init_different_prior(
        self, missing_dimension_conflict, new_dimension_conflict
    ):
        """Verify initialisation with different priors generates a side-effect conflict"""
        assert not missing_dimension_conflict.is_resolved
        assert not new_dimension_conflict.is_resolved
        resolution = missing_dimension_conflict.RenameDimensionResolution(
            missing_dimension_conflict, new_dimension_conflict
        )
        assert missing_dimension_conflict.is_resolved
        assert new_dimension_conflict.is_resolved
        assert len(resolution.new_conflicts) == 1
        assert isinstance(
            resolution.new_conflicts[0], conflicts.ChangedDimensionConflict
        )
        assert resolution.new_conflicts[0].old_prior == missing_dimension_conflict.prior
        assert resolution.new_conflicts[0].new_prior == new_dimension_conflict.prior

    def test_prefix(self, rename_dimension_resolution):
        """Verify prefix of resolution with corresponding marker"""
        assert rename_dimension_resolution.prefix == "missing~>"

    def test_repr(self, rename_dimension_resolution):
        """Verify resolution representation for user interface"""
        assert repr(rename_dimension_resolution) == "missing~>new"

    def test_adapters(self, rename_dimension_resolution):
        """Verify adapters with old and new names"""
        old_name = "missing"
        new_name = "new"
        resolution_adapters = rename_dimension_resolution.get_adapters()
        assert len(resolution_adapters) == 1
        assert (
            resolution_adapters[0].configuration
            == adapters.DimensionRenaming(old_name, new_name).configuration
        )

    def test_revert_same_prior(
        self, missing_dimension_conflict, new_dimension_same_prior_conflict
    ):
        """Verify reverting resolution set conflict to unresolved"""
        resolution = missing_dimension_conflict.RenameDimensionResolution(
            missing_dimension_conflict, new_dimension_same_prior_conflict
        )
        assert missing_dimension_conflict.is_resolved
        assert new_dimension_same_prior_conflict.is_resolved
        assert len(resolution.new_conflicts) == 0
        assert resolution.revert() == []
        assert len(resolution.new_conflicts) == 0
        assert not missing_dimension_conflict.is_resolved
        assert not new_dimension_same_prior_conflict.is_resolved
        assert missing_dimension_conflict.resolution is None
        assert new_dimension_same_prior_conflict.resolution is None

    def test_revert_different_prior(
        self,
        missing_dimension_conflict,
        new_dimension_conflict,
        rename_dimension_resolution,
    ):
        """Verify reverting resolution set conflict to unresolved and deprecate the side-effect
        conflict
        """
        assert missing_dimension_conflict.is_resolved
        assert new_dimension_conflict.is_resolved
        assert len(rename_dimension_resolution.new_conflicts) == 1
        new_conflicts = rename_dimension_resolution.new_conflicts
        assert rename_dimension_resolution.revert() == new_conflicts
        assert len(rename_dimension_resolution.new_conflicts) == 0
        assert new_conflicts[0].is_resolved
        assert not missing_dimension_conflict.is_resolved
        assert not new_dimension_conflict.is_resolved
        assert missing_dimension_conflict.resolution is None
        assert new_dimension_conflict.resolution is None


class TestAlgorithmResolution(object):
    """Test methods for resolution of algorithm changes"""

    def test_adapters(self, algorithm_resolution):
        """Verify shallow adapters for algorithm change"""
        resolution_adapters = algorithm_resolution.get_adapters()
        assert len(resolution_adapters) == 1
        assert (
            resolution_adapters[0].configuration
            == adapters.AlgorithmChange().configuration
        )

    def test_repr(self, algorithm_resolution):
        """Verify resolution representation for user interface"""
        assert repr(algorithm_resolution) == "--algorithm-change"

    def test_revert(self, algorithm_conflict, algorithm_resolution):
        """Verify reverting resolution set conflict to unresolved"""
        assert algorithm_conflict.is_resolved
        assert algorithm_resolution.revert() == []
        assert not algorithm_conflict.is_resolved
        assert algorithm_conflict.resolution is None


class TestOrionVersionResolution(object):
    """Test methods for resolution of orion version changes"""

    def test_adapters(self, orion_version_resolution):
        """Verify shallow adapters for orion version change"""
        resolution_adapters = orion_version_resolution.get_adapters()
        assert len(resolution_adapters) == 1
        assert (
            resolution_adapters[0].configuration
            == adapters.OrionVersionChange().configuration
        )

    def test_repr(self, orion_version_resolution):
        """Verify resolution representation for user interface"""
        assert repr(orion_version_resolution) == "--orion-version-change"

    def test_revert(self, orion_version_conflict, orion_version_resolution):
        """Verify reverting resolution set conflict to unresolved"""
        assert orion_version_conflict.is_resolved
        assert orion_version_resolution.revert() == []
        assert not orion_version_conflict.is_resolved
        assert orion_version_conflict.resolution is None


class TestCodeResolution(object):
    """Test methods for resolution of code conflict"""

    def test_wrong_input(self, code_conflict):
        """Verify initialization fails with invalid change type"""
        with pytest.raises(ValueError) as exc:
            code_conflict.CodeResolution(code_conflict, "yabadabadoo")
        assert "Invalid code change type" in str(exc.value)

    def test_adapters(self, code_conflict):
        """Verify adapters with code change types"""
        for change_type in adapters.CodeChange.types:
            code_resolution = code_conflict.CodeResolution(code_conflict, change_type)
            resolution_adapters = code_resolution.get_adapters()
            assert len(resolution_adapters) == 1
            assert (
                resolution_adapters[0].configuration
                == adapters.CodeChange(change_type).configuration
            )

    def test_repr(self, code_resolution):
        """Verify resolution representation for user interface"""
        assert repr(code_resolution) == "--code-change-type break"

    def test_revert(self, code_conflict, code_resolution):
        """Verify reverting resolution set conflict to unresolved"""
        assert code_conflict.is_resolved
        assert code_resolution.revert() == []
        assert not code_conflict.is_resolved
        assert code_conflict.resolution is None


class TestExperimentNameResolution(object):
    """Test methods for resolution of experiment name conflict"""

    def test_adapters(self, experiment_name_resolution):
        """Verify there is no adapters for experiment name resolution"""
        assert experiment_name_resolution.get_adapters() == []

    def test_repr(self, experiment_name_resolution):
        """Verify resolution representation for user interface"""
        assert repr(experiment_name_resolution) == "--branch-to new-exp-name"

    def test_revert(
        self,
        old_config,
        new_config,
        experiment_name_conflict,
        experiment_name_resolution,
    ):
        """Verify reverting resolution set conflict to unresolved and reset name in config"""
        assert experiment_name_conflict.is_resolved
        assert new_config["name"] == experiment_name_resolution.new_name
        assert new_config["version"] == experiment_name_resolution.new_version

        assert experiment_name_resolution.revert() == []

        assert new_config["name"] == old_config["name"]
        assert new_config["version"] == old_config["version"]
        assert not experiment_name_conflict.is_resolved
        assert experiment_name_conflict.resolution is None
