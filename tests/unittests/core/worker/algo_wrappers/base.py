from __future__ import annotations

from typing import ClassVar

import pytest

from orion.algo.space import Space
from orion.core.worker.algo_wrappers.algo_wrapper import AlgoWrapper
from orion.testing.dummy_algo import FixedSuggestionAlgo


class AlgoWrapperTests:
    """Tests for AlgoWrapper subclasses in general.

    NOTE: This is a prime candidate for a generic class, but we can't do that in py37 otherwise
    pytest complains about the test class having a __new__.
    """

    Wrapper: ClassVar[type[AlgoWrapper]]

    @pytest.fixture
    def algo_wrapper(self, space: Space):
        """Fixture that creates a Wrapper of the type `self.Wrapper` around a FixedSuggestionAlgo."""
        wrapped_space = self.Wrapper.transform_space(space)
        wrapper = self.Wrapper(space, FixedSuggestionAlgo(wrapped_space))
        return wrapper

    def test_unwrap(self, algo_wrapper: AlgoWrapper[FixedSuggestionAlgo]):
        """Test the `unwrap` method."""
        assert algo_wrapper.unwrap(type(algo_wrapper)) is algo_wrapper
        assert (
            algo_wrapper.unwrap(type(algo_wrapper.algorithm)) is algo_wrapper.algorithm
        )
        assert (
            algo_wrapper.unwrap(type(algo_wrapper.unwrapped)) is algo_wrapper.unwrapped
        )

        class _FooWrapper(AlgoWrapper):
            ...

        with pytest.raises(
            RuntimeError, match="Unable to find a wrapper or algorithm of type"
        ):
            algo_wrapper.unwrap(_FooWrapper)

    def test_repr(self, algo_wrapper: AlgoWrapper[AlgoWrapper[FixedSuggestionAlgo]]):
        """Test the `__repr__` method contains the `repr` of the wrapped algo."""

        assert str(algo_wrapper.algorithm) in str(algo_wrapper)
        assert str(algo_wrapper.unwrapped) in str(algo_wrapper)
