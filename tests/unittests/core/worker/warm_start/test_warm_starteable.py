from __future__ import annotations

from typing import Any

import pytest

from orion.algo.random import Random
from orion.core.worker.trial import Trial
from orion.core.worker.warm_start import MultiTaskWrapper
from orion.core.worker.warm_start.warm_starteable import (
    ExperimentConfig,
    WarmStarteable,
    is_warmstarteable,
)


class Foo(Random, WarmStarteable):
    def warm_start(
        self, warm_start_trials: list[tuple[ExperimentConfig, list[Trial]]]
    ) -> None:
        pass


@pytest.mark.parametrize(
    "algo_expected",
    [
        (WarmStarteable, True),
        (Random, False),
        ("random", False),
        (MultiTaskWrapper, True),
        (Foo, True),
        ("Foo", True),
        ("foo", True),
    ],
)
def test_is_warmstarteable(algo_expected: tuple[Any, bool]):
    algo, expected = algo_expected
    assert is_warmstarteable(algo) == expected
