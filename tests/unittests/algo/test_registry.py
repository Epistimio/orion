import copy

import pytest

from orion.algo.registry import Registry, RegistryMapping
from orion.algo.space import Space
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.transformer import TransformedSpace, build_required_space
from orion.core.worker.trial import Trial


@pytest.fixture
def space():
    return SpaceBuilder().build(
        {
            "x": "uniform(0, 100)",
            "y": "uniform(0, 10, discrete=True)",
            "z": 'choices(["a", "b", 0, True])',
            "f": "fidelity(1, 100, base=1)",
        }
    )


@pytest.fixture
def transformed_space(space: Space):
    return build_required_space(
        space,
        type_requirement="real",
        shape_requirement="flattened",
        dist_requirement="linear",
    )


class TestRegistry:
    """Tests for the `Registry` class. The `Registry` should basically act as a `Container[Trial]`,
    and so the tests should be roughly the same as if they were testing the built-in `dict` class.
    """

    def test_init(self):
        """Test that a new registry without trials acts as an empty container."""
        registry = Registry()
        assert len(registry) == 0
        assert not registry

    def test_register(self, space: Space):
        """Tests that appending a trial to a registry works as expected."""
        registry = Registry()
        trial = space.sample(1)[0]
        registered_id = registry.register(trial)
        assert len(registry) == 1
        assert list(registry) == [trial]

        assert registry[registered_id] == trial

    def test_register_overwrite_with_results(self, space: Space):
        """Tests that registering a trial with the same params overwrites the existing trial."""
        registry = Registry()
        trial = space.sample(1)[0]
        registered_id = registry.register(trial)
        assert len(registry) == 1
        assert list(registry) == [trial]

        assert registry[registered_id] == trial

        same_but_with_results = copy.deepcopy(trial)
        same_but_with_results._results.append(
            Trial.Result(name="objective", type="objective", value=1)
        )

        same_id = registry.register(same_but_with_results)
        assert same_id == registered_id
        assert len(registry) == 1
        assert list(registry) == [same_but_with_results]

    @pytest.mark.parametrize("status", ["completed", "interrupted", "failed", "broken"])
    def test_register_overwrite_with_status(self, space: Space, status: str):
        """Tests that registering a trial with the same params overwrites the existing trial."""
        registry = Registry()
        trial = space.sample(1)[0]
        registered_id = registry.register(trial)
        assert len(registry) == 1
        assert list(registry) == [trial]

        assert registry[registered_id] == trial

        same_but_with_status = copy.deepcopy(trial)
        same_but_with_status._status = status

        same_id = registry.register(same_but_with_status)
        assert same_id == registered_id
        assert len(registry) == 1
        assert list(registry) == [same_but_with_status]

    def test_register_overwrite_with_experiment(self, space: Space):
        """Tests that registering a trial with the same params overwrites the existing trial."""
        registry = Registry()
        trial = space.sample(1)[0]
        registered_id = registry.register(trial)
        assert len(registry) == 1
        assert list(registry) == [trial]

        assert registry[registered_id] == trial
        experiment = "BLABLABOB"  # TODO: Use an experiment fixture of some sort.
        same_but_with_experiment = copy.deepcopy(trial)
        same_but_with_experiment.experiment = experiment

        same_id = registry.register(same_but_with_experiment)
        assert same_id == registered_id
        assert len(registry) == 1
        assert list(registry) == [same_but_with_experiment]


class TestRegistryMapping:
    """Tests for the `RegistryMapping` class. The RegistryMapping should act as a
    Mapping[Trial, list[Trial]].
    """

    def test_init(self):
        """Tests that a new RegistryMapping acts as an empty dict."""
        original = Registry()
        transformed = Registry()
        mapping = RegistryMapping(
            original_registry=original, transformed_registry=transformed
        )
        assert not mapping
        assert len(mapping) == 0
        assert not mapping.keys()
        assert not mapping.values()
        assert not mapping.items()

    def test_extenal_register_doesnt_increase_len(
        self, space: Space, transformed_space: TransformedSpace
    ):
        """Test that externally registering trials in the original or transformed registries does
        not affect the length of the mapping.
        """
        original = Registry()
        transformed = Registry()
        mapping = RegistryMapping(
            original_registry=original, transformed_registry=transformed
        )
        assert not mapping
        assert len(mapping) == 0

        original_trial = space.sample(1)[0]
        original.register(original_trial)
        assert not mapping

        transformed_trial = transformed_space.sample(1)[0]
        transformed.register(transformed_trial)
        assert not mapping

    def test_register(self, space: Space, transformed_space: TransformedSpace):
        """Tests for the `register` method of the `RegistryMapping` class."""
        original_reg = Registry()
        transformed_reg = Registry()
        mapping = RegistryMapping(
            original_registry=original_reg, transformed_registry=transformed_reg
        )

        original_trial = space.sample(1)[0]
        transformed_trial = transformed_space.transform(original_trial)

        mapping.register(original_trial, transformed_trial)
        # NOTE: register doesn't actually register the trial, it just adds it to the mapping.
        assert len(mapping) == 1
        assert original_trial in mapping

        # NOTE: Here since we assume that the trials are supposed to be registered in the registries
        # externally, we can't yet iterate over the mapping (e.g. with keys(), values() or items()).

        # Now we actually register the trials in the individual registries.
        assert original_trial not in original_reg
        original_stored_id = original_reg.register(original_trial)
        assert transformed_trial not in transformed_reg
        transformed_stored_id = transformed_reg.register(transformed_trial)

        assert mapping._mapping == {original_stored_id: {transformed_stored_id}}
        assert list(mapping.keys()) == [original_trial]
        assert list(mapping.values()) == [[transformed_trial]]
        assert mapping[original_trial] == [transformed_trial]
