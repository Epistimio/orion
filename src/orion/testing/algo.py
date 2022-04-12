# -*- coding: utf-8 -*-
"""Generic tests for Algorithms"""
from __future__ import annotations

import copy
import functools
import inspect
import itertools
import logging
from collections import defaultdict
from typing import Callable, ClassVar, NamedTuple, Type, TypeVar, Sequence

import numpy
import pytest

import orion.algo.base
from orion.algo.asha import ASHA
from orion.algo.base import BaseAlgorithm
from orion.algo.gridsearch import GridSearch
from orion.algo.hyperband import Hyperband
from orion.algo.parallel_strategy import strategy_factory
from orion.algo.random import Random
from orion.algo.space import Space
from orion.algo.tpe import TPE
from orion.benchmark.task.branin import Branin
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils import backward, format_trials
from orion.core.worker.primary_algo import SpaceTransformAlgoWrapper, create_algo
from orion.core.worker.transformer import build_required_space
from orion.core.worker.trial import Trial
from orion.testing.space import build_space


def phase(some_test):
    return pytest.mark.usefixtures("phase")(some_test)


def customized_mutate_example(search_space, rng, old_value, **kwargs):
    """Define a customized mutate function example"""
    multiply_factor = kwargs.pop("multiply_factor", 3.0)
    add_factor = kwargs.pop("add_factor", 1)
    if search_space.type == "real":
        new_value = old_value / multiply_factor
    elif search_space.type == "integer":
        new_value = int(old_value + add_factor)
    else:
        new_value = old_value
    return new_value


class TestPhase(NamedTuple):
    name: str
    """ Name of the test phase."""

    n_trials: int
    """ Number of trials after which the phase should begin."""

    method_to_spy: str
    """ Name of the algorithm's attribute to use to spy.
    NOTE: need to clarify what exactly this is used for.
    """

    # just so pytest doesn't complain about this.
    __test__ = False  # type: ignore


class BaseAlgoTests:
    """Generic Test-suite for HPO algorithms.

    This test-suite covers all typical cases for HPO algorithms. To use it for a new algorithm,
    the class inheriting from this one must redefine the attributes ``algo_name`` with
    the name of the algorithm used to create it with the algorithm factory
    ``orion.core.worker.primary_algo.SpaceTransformAlgoWrapper`` and ``config`` with a base
    configuration for the algorithm that contains all its arguments. The base space can be redefine
    if needed with the attribute ``space``.

    Many algorithms have different phases that should be tested. For instance
    TPE have a first phase of random search and a second of Bayesian Optimization.
    The random search and Bayesian optimization are different implementations and both should be
    tested. For this reason, the class method ``orion.testing.algo.BaseAlgoTests.set_phases`` must
    be called to parametrize the tests with phases. Failure to doing so will causes the tests to
    crash. See ``tests/unittests/algo/test_tpe.py`` for an example.
    """

    algo_name = None
    config = {}
    max_trials = 200
    space = {"x": "uniform(0, 1)", "y": "uniform(0, 1)"}

    phases: ClassVar[list[TestPhase]]
    _current_phase: ClassVar[TestPhase]

    def __init_subclass__(cls) -> None:
        # cls.set_phases(cls.phases)
        if not hasattr(cls, "phases") or not cls.phases:
            raise ValueError(
                f"Test classes MUST set a value for the `phases` class attribute."
            )
        from itertools import accumulate

        cls._current_phase = cls.phases[0]

        @pytest.fixture(
            autouse=False,  # todo; not sure if this should always be true by default.
            scope="module",
            params=cls.phases,
            ids=[phase.name for phase in cls.phases],
        )
        def phase(request):
            test_phase: TestPhase = request.param
            # Temporarily change the class attribute holding the current phase.
            original_phase = cls._current_phase
            cls._current_phase = test_phase
            yield test_phase
            cls._current_phase = original_phase

        cls.phase = staticmethod(phase)

    @classmethod
    def set_phases(cls, phases: Sequence[TestPhase]):
        """Parametrize the tests with different phases.

        Some algorithms have different phases that should be tested. For instance
        TPE have a first phase of random search and a second of Bayesian Optimization.
        The random search and Bayesian optimization are different implementations and both should be
        tested.

        Parameters
        ----------
        phases: list of tuples
            The different phases to test. The format of the tuples should be
            (str(id of the test),
            int(number of trials before the phase begins),
            str(name of the algorithm's attribute to spy (ex: "space.sample"))
            )
        """
        cls.phases = [TestPhase(*phase) for phase in phases]

    @classmethod
    def create_algo(
        cls,
        config: dict | None = None,
        space: Space | None = None,
        seed: int | Sequence[int] | None = None,
        **kwargs,
    ):
        """Create the algorithm based on config.

        Also initializes the algorithm with the required number of random trials from the previous
        test phases before returning it.

        If `seed` is passed, then `seed_rng` is called before observing anything.

        Parameters
        ----------
        config: dict, optional
            The configuration for the algorithm. ``cls.config`` will be used
            if ``config`` is ``None``.
        space: ``orion.algo.space.Space``, optional
            Space object to pass to algo. The output of ``cls.create_space()``
            will be used if ``space`` is ``None``.
        kwargs: dict
            Values to override algorithm configuration.
        """
        config = copy.deepcopy(config or cls.config)
        config.update(kwargs)
        if hasattr(cls, "algo_type"):
            base_algo_type = cls.algo_type
        else:
            base_algo_type = orion.algo.base.algo_factory.get_class(cls.algo_name)

        original_space = space or cls.create_space()
        algo = create_algo(space=original_space, algo_type=base_algo_type, **config)
        algo.algorithm.max_trials = cls.max_trials

        # NOTE: Should this be called before, or after the rest?
        if seed is not None:
            algo.seed_rng(seed)

        n_previous_trials = cls._current_phase.n_trials
        if n_previous_trials >= cls.max_trials:
            raise ValueError(
                f"Test isn't configured properly: max_trials ({cls.max_trials}) is larger than "
                f"the total number of trials seen so far when in phase ({cls._current_phase}). "
                f"Increasing max_trials might be a good idea. "
            )

        # Force the algo to observe the given number of trials.
        cls.force_observe(n_previous_trials, algo)
        # TODO: Should we check that the algo has indeed observed the right number of trials that we
        # want, and that the max_trials hasn't been busted, if present?
        assert algo.n_observed == n_previous_trials
        return algo

    def update_space(self, test_space):
        """Get complete space configuration with partial overwrite

        The values passed in ``test_space`` will override the default values
        in ``self.config``.

        Parameters
        ----------
        test_space: dic
            The configuration for the space.
        """
        space = copy.deepcopy(self.space)
        space.update(test_space)
        return space

    @classmethod
    def create_space(cls, space: dict | None = None):
        """Create the space object

        Parameters
        ----------
        space: dict, optional
            Configuration of the search space. The default ``self.space`` will be used
            if ``space`` is ``None``.

        """
        return SpaceBuilder().build(space if space is not None else cls.space)

    @classmethod
    def observe_trials(
        cls, trials: list[Trial], algo: BaseAlgorithm, objective: float = 0
    ):
        """Make the algorithm observe trials

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
            Trials formatted as tuples of values
        algo: ``orion.algo.base.BaseAlgorithm``
            The algorithm used to observe trials.
        objective: int, optional
            The base objective for the trials. All objectives
            will have value ``objective + i``. Defaults to 0.
        """
        backward.algo_observe(
            algo, trials, [dict(objective=objective + i) for i in range(len(trials))]
        )

    @classmethod
    def get_num(cls, num: int):
        """Force number of trials to suggest

        Some algorithms must be tested with specific number of suggests at a time (ex: ASHA).
        This method can be overriden to change ``num`` based on the special needs.
        """
        return num

    @classmethod
    def force_observe(cls, num: int, algo: BaseAlgorithm):
        """Force observe ``num`` trials.

        Parameters
        ----------
        num: int
            Number of trials to suggest and observe.
        algo: ``orion.algo.base.BaseAlgorithm``
            The algorithm that must suggest and observe.

        Raises
        ------
        RuntimeError
            - If the algorithm returns duplicates. Algorithms may return duplicates across workers,
              but in sequential scenarios as here, it should not happen.
            - If the algorithm fails to sample any trial at least 5 times.
        """
        objective = 0
        failed = 0
        MAX_FAILED = 5
        ids = set()

        while not algo.is_done and algo.n_observed < num and failed < MAX_FAILED:
            trials = algo.suggest(cls.get_num(num - algo.n_observed))
            if len(trials) == 0:
                failed += 1
                continue
            for trial in trials:
                if trial.hash_name in ids:
                    raise RuntimeError(f"algo suggested a duplicate: {trial}")
                ids.add(trial.hash_name)
            cls.observe_trials(trials, algo, objective)
            objective += len(trials)

        if failed >= MAX_FAILED:
            raise RuntimeError(
                f"Algorithm cannot sample more than {algo.n_observed} trials. Is it normal?"
            )

    def spy_phase(self, mocker, num, algo, attribute):
        """Force observe ``num`` trials and then mock a given method to count calls.

        Parameters
        ----------
        mocker: ``pytest_mock.mocker``
            Mocker from ``pytest_mock``. Should be given by fixtures of the tests.
        num: int
            Number of trials to suggest and observe
        algo: ``orion.algo.base.BaseAlgorithm``
            The algorithm to test
        attribute: str
            The algorithm attribute or method to mock. The path is respective to the
            algorithm object. For example, a valid value would be `'space.sample'`
            which will mock ``algo.algorithm.space.sample``.
        """
        self.force_observe(num, algo)
        spy = spy_attr(mocker, algo.algorithm, attribute)
        return spy

    def assert_callbacks(self, spy, num, algo):
        """Callback to make special asserts at end of tests

        Override this method in algorithm test-suite to customize verifications done at end
        of tests.

        Parameters
        ----------
        spy: Mocked object
            Object mocked by ``BaseAlgoTests.spy_phase``.
        num: int
            number of trials of the phase.
        algo: ``orion.algo.base.BaseAlgorithm``
            The algorithm being tested.
        """
        pass

    def assert_dim_type_supported(self, test_space: dict):
        """Test that a given dimension type is properly supported by the algorithm

        This will test that the algorithm sample trials valid for the given type
        and that the algorithm can observe these trials.

        Parameters
        ----------
        test_space: the search space of the test.
        """
        space = self.create_space(self.update_space(test_space))
        algo = self.create_algo(space=space)

        trials = algo.suggest(1)
        assert trials[0] in space
        self.observe_trials(trials, algo, 1)

    def test_configuration(self):
        """Test that configuration property attribute contains all class arguments."""
        algo = self.create_algo()
        assert algo.configuration != self.create_algo(config={})
        assert algo.configuration == {self.algo_name: self.config}

    def test_get_id(self):
        """Test that the id hashing is valid"""
        space = self.create_space(
            space=self.update_space({"f": "fidelity(1, 10, base=2)"})
        )

        algo = self.create_algo(space=space)

        def get_id(point, ignore_fidelity=False, exp_id=None):
            trial = format_trials.tuple_to_trial(point, space)
            trial.experiment = exp_id
            return algo.get_id(
                trial,
                ignore_fidelity=ignore_fidelity,
            )

        assert get_id([1, 1, 1]) == get_id([1, 1, 1])
        assert get_id([1, 1, 1]) != get_id([1, 2, 2])
        assert get_id([1, 1, 1]) != get_id([2, 1, 1])

        assert get_id([1, 1, 1], ignore_fidelity=False) == get_id(
            [1, 1, 1], ignore_fidelity=False
        )
        # Fidelity changes id
        assert get_id([1, 1, 1], ignore_fidelity=False) != get_id(
            [2, 1, 1], ignore_fidelity=False
        )
        # Non-fidelity changes id
        assert get_id([1, 1, 1], ignore_fidelity=False) != get_id(
            [1, 1, 2], ignore_fidelity=False
        )

        assert get_id([1, 1, 1], ignore_fidelity=True) == get_id(
            [1, 1, 1], ignore_fidelity=True
        )
        # Fidelity does not change id
        assert get_id([1, 1, 1], ignore_fidelity=True) == get_id(
            [2, 1, 1], ignore_fidelity=True
        )
        # Non-fidelity still changes id
        assert get_id([1, 1, 1], ignore_fidelity=True) != get_id(
            [1, 1, 2], ignore_fidelity=True
        )

        # Experiment id is ignored
        assert get_id([1, 1, 1], exp_id=1) == get_id([1, 1, 1], exp_id=2)

    @phase
    @pytest.mark.parametrize("seed", [123])
    def test_seed_rng(self, seed: int):
        """Test that the seeding gives reproducibile results."""
        algo = self.create_algo(seed=seed)

        trial_a = algo.suggest(1)[0]
        trial_b = algo.suggest(1)[0]
        assert trial_a.id != trial_b.id

        new_algo = self.create_algo(seed=seed)
        assert new_algo.n_observed == algo.n_observed
        trial_c = new_algo.suggest(1)[0]
        assert trial_c.id == trial_a.id

    @phase
    def test_seed_rng_init(self):
        """Test that the seeding gives reproducibile results."""
        algo = self.create_algo(seed=1)
        trials = algo.suggest(1)
        assert trials is not None
        assert algo.suggest(1)[0].id != trials[0].id

        new_algo = self.create_algo(seed=2)
        self.force_observe(algo.n_observed, new_algo)
        assert new_algo.suggest(1)[0].id != trials[0].id

        new_algo = self.create_algo(seed=1)
        self.force_observe(algo.n_observed, new_algo)
        assert new_algo.suggest(1)[0].id == trials[0].id

    @phase
    @pytest.mark.parametrize("seed", [numpy.random.randint(10000)])
    def test_state_dict(self, seed: int):
        """Verify that resetting state makes sampling deterministic"""
        algo = self.create_algo(seed=seed)

        state = algo.state_dict
        a = algo.suggest(1)[0]

        new_algo = self.create_algo()
        assert a.id != new_algo.suggest(1)[0].id

        new_algo.set_state(state)
        assert a.id == new_algo.suggest(1)[0].id

    @phase
    def test_suggest_n(self):
        """Verify that suggest returns correct number of trials if ``num`` is specified in ``suggest``."""
        algo = self.create_algo()
        trials = algo.suggest(5)
        assert trials is not None
        assert len(trials) == 5

    @phase
    def test_has_suggested(self):
        """Verify that algorithm detects correctly if a trial was suggested"""
        algo = self.create_algo()
        a = algo.suggest(1)[0]
        assert algo.has_suggested(a)
        # NOTE: not algo.has_suggested(some random trial) is tested in test_has_suggested_statedict

    @phase
    def test_has_suggested_statedict(self):
        """Verify that algorithm detects correctly if a trial was suggested even when state was restored."""
        algo = self.create_algo()

        a = algo.suggest(1)[0]
        state = algo.state_dict
        assert algo.has_suggested(a)

        algo = self.create_algo()
        assert not algo.has_suggested(a)

        algo.set_state(state)
        assert algo.has_suggested(a)

    @phase
    def test_observe(self):
        """Verify that algorithm observes trial without any issues"""
        algo = self.create_algo()

        a = algo.space.sample()[0]
        backward.algo_observe(algo, [a], [dict(objective=1)])

        b = algo.suggest(1)[0]
        backward.algo_observe(algo, [b], [dict(objective=2)])

    @phase
    def test_has_observed(self):
        """Verify that algorithm detects correctly if a trial was observed"""
        algo = self.create_algo()

        a = algo.suggest(1)[0]
        assert not algo.has_observed(a)
        backward.algo_observe(algo, [a], [dict(objective=1)])
        assert algo.has_observed(a)

        b = algo.suggest(1)[0]
        assert not algo.has_observed(b)
        backward.algo_observe(algo, [b], [dict(objective=2)])
        assert algo.has_observed(b)

    @phase
    def test_has_observed_statedict(self):
        """Verify that algorithm detects correctly if a trial was observed even when state was restored."""
        algo = self.create_algo()

        a = algo.suggest(1)[0]
        backward.algo_observe(algo, [a], [dict(objective=1)])
        state = algo.state_dict

        algo = self.create_algo()
        assert not algo.has_observed(a)
        algo.set_state(state)
        assert algo.has_observed(a)

        b = algo.suggest(1)[0]
        backward.algo_observe(algo, [b], [dict(objective=2)])
        state = algo.state_dict

        algo = self.create_algo()
        assert not algo.has_observed(b)
        algo.set_state(state)
        assert algo.has_observed(b)

    @phase
    def test_n_suggested(self):
        """Verify that algorithm returns correct number of suggested trials"""
        algo = self.create_algo()
        initial = algo.n_suggested
        algo.suggest(1)
        assert algo.n_suggested == initial + 1

    def test_n_observed(self, phase: list[TestPhase]):
        """Verify that algorithm returns correct number of observed trials"""
        algo = self.create_algo()
        initial = algo.n_observed
        trials = algo.suggest(1)
        assert algo.n_observed == initial
        assert len(trials) == 1
        self.observe_trials(trials, algo)
        assert algo.n_observed == initial + 1

    @phase
    def test_real_data(self):
        """Test that algorithm supports real dimesions"""
        self.assert_dim_type_supported({"x": "uniform(0, 5)"})

    @phase
    def test_int_data(self):
        """Test that algorithm supports integer dimesions"""
        self.assert_dim_type_supported({"x": "uniform(0, 5000, discrete=True)"})

    @phase
    def test_cat_data(self):
        """Test that algorithm supports categorical dimesions"""
        self.assert_dim_type_supported(
            {  # Add 3 dims so that there exists many possible trials for the test
                "x": "choices(['a', 0.2, 1, None])",
                "y": "choices(['a', 0.2, 1, None])",
                "z": "choices(['a', 0.2, 1, None])",
            },
        )

    @phase
    def test_logreal_data(self):
        """Test that algorithm supports logreal dimesions"""
        self.assert_dim_type_supported({"x": "loguniform(1, 5)"})

    @phase
    def test_logint_data(self):
        """Test that algorithm supports loginteger dimesions"""
        self.assert_dim_type_supported({"x": "loguniform(1, 100, discrete=True)"})

    @phase
    def test_shape_data(self):
        """Test that algorithm supports dimesions with shape"""
        self.assert_dim_type_supported({"x": "uniform(0, 5, shape=(3, 2))"})

    @phase
    def test_broken_trials(self):
        """Test that algorithm can handle broken trials"""
        algo = self.create_algo()
        trial = algo.suggest(1)[0]
        trial.status = "broken"
        assert not algo.has_observed(trial)
        algo.observe([trial])
        assert algo.has_observed(trial)

    def test_is_done_cardinality(self):
        """Test that algorithm will stop when cardinality is reached"""
        space = SpaceBuilder().build(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "loguniform(1, 6, discrete=True)",
            }
        )
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        i = 0
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = algo.n_suggested
            backward.algo_observe(
                algo,
                [format_trials.tuple_to_trial([x, y, z], space)],
                [dict(objective=i)],
            )
            assert algo.n_suggested == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done

    def test_is_done_max_trials(self):
        """Test that algorithm will stop when max trials is reached"""
        algo = self.create_algo()
        self.force_observe(self.max_trials, algo)

        assert algo.is_done

    def test_optimize_branin(self):
        """Test that algorithm optimizes somehow (this is on-par with random search)"""
        MAX_TRIALS = 20
        task = Branin()
        space = self.create_space(task.get_search_space())
        algo = self.create_algo(config={}, space=space)
        algo.algorithm.max_trials = MAX_TRIALS
        safe_guard = 0
        trials = []
        objectives = []
        while trials or not algo.is_done:
            if safe_guard >= MAX_TRIALS:
                break

            if not trials:
                trials = algo.suggest(MAX_TRIALS - len(objectives))

            trial = trials.pop(0)
            results = task(trial.params["x"])
            objectives.append(results[0]["value"])
            backward.algo_observe(algo, [trial], [dict(objective=objectives[-1])])
            safe_guard += 1

        assert algo.is_done
        assert min(objectives) <= 10


class BaseParallelStrategyTests:
    """Generic Test-suite for parallel strategies.

    This test-suite follow the same logic than  BaseAlgoTests, but applied for ParallelStrategy
    classes.
    """

    parallel_strategy_name = None
    config = {}
    expected_value = None
    default_value = None

    def create_strategy(self, config=None, **kwargs):
        """Create the parallel strategy based on config.

        Parameters
        ----------
        config: dict, optional
            The configuration for the parallel strategy. ``self.config`` will be used
            if ``config`` is ``None``.
        kwargs: dict
            Values to override strategy configuration.
        """
        config = copy.deepcopy(config or self.config)
        config.update(kwargs)
        return strategy_factory.create(**self.config)

    def get_trials(self):
        """10 objective observations"""
        trials = []
        for i in range(10):
            trials.append(
                Trial(
                    params=[{"name": "x", "type": "real", "value": i}],
                    results=[{"name": "objective", "type": "objective", "value": i}],
                    status="completed",
                )
            )

        return trials

    def get_noncompleted_trial(self, status="reserved"):
        """Return a single trial without results"""
        return Trial(
            params=[{"name": "a", "type": "integer", "value": 6}], status=status
        )

    def get_corrupted_trial(self):
        """Return a corrupted trial with results but status reserved"""
        return Trial(
            params=[{"name": "a", "type": "integer", "value": 6}],
            results=[{"name": "objective", "type": "objective", "value": 1}],
            status="reserved",
        )

    def test_configuration(self):
        """Test that configuration property attribute contains all class arguments."""
        strategy = self.create_strategy()
        assert strategy.configuration != self.create_strategy(config={})
        assert strategy.configuration == self.config

    def test_state_dict(self):
        """Verify state is restored properly"""
        strategy = self.create_strategy()

        strategy.observe(self.get_trials())

        new_strategy = self.create_strategy()
        assert strategy.state_dict != new_strategy.state_dict

        new_strategy.set_state(strategy.state_dict)
        assert strategy.state_dict == new_strategy.state_dict

        noncompleted_trial = self.get_noncompleted_trial()

        if strategy.infer(noncompleted_trial) is None:
            assert strategy.infer(noncompleted_trial) == new_strategy.infer(
                noncompleted_trial
            )
        else:
            assert (
                strategy.infer(noncompleted_trial).objective.value
                == new_strategy.infer(noncompleted_trial).objective.value
            )

    def test_infer_no_history(self):
        """Test that strategy can infer even without having seen trials"""
        noncompleted_trial = self.get_noncompleted_trial()
        trial = self.create_strategy().infer(noncompleted_trial)
        if self.expected_value is None:
            assert trial is None
        elif self.default_value is None:
            assert trial.objective.value == self.expected_value
        else:
            assert trial.objective.value == self.default_value

    def test_handle_corrupted_trials(self, caplog):
        """Test that strategy can handle trials that has objective but status is not
        properly set to completed."""
        corrupted_trial = self.get_corrupted_trial()
        with caplog.at_level(logging.WARNING, logger="orion.algo.parallel_strategy"):
            trial = self.create_strategy().infer(corrupted_trial)

        match = "Trial `{}` has an objective but status is not completed".format(
            corrupted_trial.id
        )
        assert match in caplog.text

        assert trial is not None
        assert trial.objective.value == corrupted_trial.objective.value

    def test_handle_noncompleted_trials(self, caplog):
        with caplog.at_level(logging.WARNING, logger="orion.algo.parallel_strategy"):
            self.create_strategy().infer(self.get_noncompleted_trial())

        assert (
            "Trial `{}` has an objective but status is not completed" not in caplog.text
        )

    def test_strategy_value(self):
        """Test that ParallelStrategy returns the expected value"""
        strategy = self.create_strategy()
        strategy.observe(self.get_trials())
        trial = strategy.infer(self.get_noncompleted_trial())

        if self.expected_value is None:
            assert trial is None
        else:
            assert trial.objective.value == self.expected_value
