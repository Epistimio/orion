"""Perform integration tests for `orion.algo.HEBO`."""
from __future__ import annotations

import dataclasses
import typing
from typing import ClassVar

import pytest
from hebo.models.model_factory import model_dict
from pymoo.factory import get_algorithm_options

from orion.algo.hebo.hebo_algo import (
    HEBO,
    EvolutionStrategyName,
    ModelName,
    properly_seeded_models,
)
from orion.testing.algo import BaseAlgoTests, TestPhase, first_phase_only

if typing.TYPE_CHECKING:
    from _pytest.fixtures import SubRequest
_model_names = sorted(model_dict.keys())
_es_names = sorted(dict(get_algorithm_options()).keys())


RUN_QUICK = True
# NOTE: Can set this to either `skip_param` or `xfail_param` to make the tests shorter or stricter.
def skip_or_xfail_mark(
    *args, reason: str, raises: type[Exception] | tuple[type[Exception], ...] = ()
):
    if raises and not RUN_QUICK:
        return pytest.mark.xfail(*args, reason=reason, raises=raises)
    return pytest.mark.skip(reason=reason)


@pytest.fixture(params=_model_names)
def model_name(request: SubRequest):
    """Fixture that yields the `model_name` parameter to pass to the Hebo constructor."""
    _model_name: ModelName = request.param
    yield _model_name


_poor_constructor = skip_or_xfail_mark(
    reason="Constructor isn't well written (should pop stuff from kwargs)",
    raises=TypeError,
)
_pymoo_error = skip_or_xfail_mark(
    reason="causes an error in pymoo code (`self.sampling` is np.ndarray)",
    raises=AttributeError,
)
_wrong_constructor_args = skip_or_xfail_mark(
    reason="ES class isn't receiving required constructor arguments",
    raises=TypeError,
)
_tournament_selection_bool_casting = skip_or_xfail_mark(
    reason="ES does bool(np.ndarray) in pymoo/operators/selection/tournament.py",
    raises=ValueError,
)
_numpy_broadcasting_issue = skip_or_xfail_mark(
    reason="ES has a numpy broadcasting issue",
    raises=ValueError,
)
evolutionary_strategy_marks: dict[EvolutionStrategyName, list[pytest.MarkDecorator]] = {  # type: ignore
    "nsga2": [],
    "de": [_poor_constructor],
    "brkga": [_poor_constructor],
    "cmaes": [_pymoo_error],
    "nelder": [_pymoo_error],
    "nelder-mead": [_pymoo_error],
    "pattern-search": [_pymoo_error],
    "ctaea": [_wrong_constructor_args],
    "moead": [_wrong_constructor_args],
    "nsga3": [_wrong_constructor_args],
    "rnsga2": [_wrong_constructor_args],
    "rnsga3": [_wrong_constructor_args],
    "unsga3": [_wrong_constructor_args],
    "ga": [_tournament_selection_bool_casting],
    "pso": [_numpy_broadcasting_issue],
}
default = skip_or_xfail_mark(reason="Isn't explicitly supported.")


@pytest.fixture(
    params=[
        pytest.param(es, marks=evolutionary_strategy_marks.get(es, default))
        for es in _es_names
    ]
)
def evolutionary_strategy(request: SubRequest):
    """Fixture that yields the `es` parameter to pass to the Hebo constructor."""
    es_name: EvolutionStrategyName = request.param
    yield es_name


@pytest.fixture(autouse=True)
def _config(model_name: ModelName, evolutionary_strategy: EvolutionStrategyName):
    """Fixture that parametrizes the configuration used in the tests below."""
    backup: HEBO.Parameters = TestHEBO.config["parameters"]
    new_params: HEBO.Parameters = dataclasses.replace(
        backup,
        model_name=model_name,
        evolutionary_strategy=evolutionary_strategy,
    )
    # Replace the value in the Config.
    TestHEBO.config["parameters"] = new_params
    yield
    TestHEBO.config["parameters"] = backup


@pytest.fixture(autouse=True)
def xfail_if_unseeded_model_chosen(request: SubRequest):
    """Adds a xfail mark on tests that relate to seeding when non-seeded model is chosen."""
    # TODO: Can't refer to the tests by reference because of `parametrize_this`: their `__name__`
    # becomes "method". If/when we rework `parametrize_this` or change the name of tests, it'll
    # be important to update this as well.
    # NOTE: Also, normally I'd add a fixture to only these tests, but that's not currently possible:
    # It seems like the signature of these tests can't be changed, because of `@phase` and/or
    # `parametrize_this`.
    tests_that_check_seeding = [
        TestHEBO.test_seed_rng,
        TestHEBO.test_seed_rng_init,
        TestHEBO.test_state_dict,
    ]

    model_name: str = request.getfixturevalue("model_name")

    if model_name in properly_seeded_models:
        return  # Don't add any mark, the test is expected to pass.

    # NOTE: We need to detect the phase. The reason for this is so we can avoid having a
    # bunch of tests XPASS when the test is ran in the random phase (where some do work).
    if "num" not in request.fixturenames:
        return  # One of the tests that doesn't involve the phase.

    in_random_phase: bool = request.getfixturevalue("num") == 0
    if in_random_phase:
        return

    # NOTE: Also can't use `request.function` because of `parametrize_this`, since it points
    # to the local closure inside `parametrize_this`.
    # if request.function in test_that_check_seeding:
    if any(func == request.function for func in tests_that_check_seeding):
        request.node.add_marker(
            pytest.mark.xfail(
                reason=f"This model name {model_name} is not properly seeded.",
            )
        )


# Number of initial random points.
N_INIT = 10


class TestHEBO(BaseAlgoTests):
    """Test suite for the HEBO algorithm."""

    algo_name: ClassVar[str] = "hebo"
    config = {
        "seed": 1234,
        "parameters": HEBO.Parameters(random_samples=N_INIT),
    }

    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("hebo", N_INIT, "suggest"),
    ]
    max_trials: ClassVar[int] = N_INIT + 10

    @first_phase_only
    def test_seed_rng_init(self):
        """Test that the seeding gives reproducible results."""
        algo = self.create_algo(seed=1)

        trials = algo.suggest(1)
        same_trials = algo.suggest(1)
        assert trials is not None
        assert same_trials is not None
        assert same_trials[0].id != trials[0].id

        new_algo = self.create_algo(seed=2)
        self.force_observe(algo.n_observed, new_algo)
        # NOTE: Removing this check from the base test, since the values generated in the random
        # search phase for HEBO are actually pretty consistent across seeds. This is due to them
        # using a `torch.quasirandom.SobolEngine`, which seems to "map out" the space, rather than
        # take random samples from it.
        # assert new_algo.suggest(1)[0].id != trials[0].id

        new_algo = self.create_algo(seed=1)
        self.force_observe(algo.n_observed, new_algo)
        same_trials = new_algo.suggest(1)
        assert same_trials is not None
        assert same_trials[0].id == trials[0].id
