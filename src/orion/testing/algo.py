"""Generic tests for Algorithms"""
from __future__ import annotations

import copy
import inspect
import itertools
import logging
from dataclasses import dataclass, field
from typing import ClassVar, Sequence, TypeVar

import numpy
import pytest

import orion.algo.base
from orion.algo.base import BaseAlgorithm
from orion.algo.parallel_strategy import strategy_factory
from orion.algo.space import Space
from orion.benchmark.task.branin import Branin
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils import backward, format_trials
from orion.core.worker.primary_algo import SpaceTransformAlgoWrapper, create_algo
from orion.core.worker.trial import Trial

AlgoType = TypeVar("AlgoType", bound=BaseAlgorithm)


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


@dataclass
class TestPhase:
    name: str
    """ Name of the test phase."""

    n_trials: int
    """ Number of trials after which the phase should begin."""

    method_to_spy: str | None = None
    """ Name of the method or function that is supposed to create the trials during that test phase.

    This is currently unused. Tests could potentially pass this as an argument to mocker.spy to
    check that the method is called the right number of times during each phase.
    """

    # The previous phase or None if this is the first one.
    prev: TestPhase | None = field(default=None, repr=False)
    # The next phase, or an int with max_trials.
    next: TestPhase | int | None = field(default=None, repr=False)

    @property
    def length(self) -> int:
        """Returns the duration of this test phase, in number of trials."""
        assert self.next
        next_start = (
            self.next.n_trials if isinstance(self.next, TestPhase) else self.next
        )
        return next_start - self.n_trials

    @property
    def end_n_trials(self) -> int:
        """Returns the end of this test phase (either start of next phase or max_trials)."""
        return self.n_trials + self.length


# just so pytest doesn't complain about this.
TestPhase.__test__ = False  # type: ignore


def _are_equal(a, b) -> bool:
    """Compare two statedicts and return if they are equal. This is required because of annoying
    numpy array comparisons and such.
    """
    try:
        numpy.testing.assert_equal(a, b)
        return True
    except AssertionError:
        return False


def first_phase_only(test):
    """Decorator to run a test only on the first phase of the algorithm."""
    return pytest.mark.usefixtures("first_phase")(test)


def last_phase_only(test):
    """Decorator to run a test only on the last test phase of the algorithm."""
    return pytest.mark.usefixtures("last_phase")(test)


# NOTE: Can't make the test class generic in python 3.7, because it adds a __new__ constructor to
# the type, which prevents it being collected.


class BaseAlgoTests:
    """Generic Test-suite for HPO algorithms.

    This test-suite covers all typical cases for HPO algorithms. To use it for a new algorithm,
    the class inheriting from this one must redefine the attributes ``algo_name`` with
    the name of the algorithm used to create it with the algorithm factory
    ``orion.core.worker.primary_algo.SpaceTransformAlgoWrapper`` and ``config`` with a base
    configuration for the algorithm that contains all its arguments. The base space can be redefine
    if needed with the attribute ``space``.

    Most algorithms have different phases that should be tested. For instance TPE has a first phase
    of random search and a second of Bayesian Optimization. The random search and Bayesian
    optimization phases use different logic and should both be tested.
    The `phases` class attribute can be set to parametrize all tests with each phase.
    See ``tests/unittests/algo/test_tpe.py`` for an example.
    """

    algo_type: type[AlgoType]
    """ The type of algorithm under test."""

    algo_name: ClassVar[str | None] = None

    config: ClassVar[dict] = {}
    space: ClassVar[dict] = {"x": "uniform(0, 1)", "y": "uniform(0, 1)"}

    phases: ClassVar[list[TestPhase]] = [TestPhase("default", 0, "sample")]
    """ Test phases for the algorithms. Overwrte this if the algorithm has more than one phase."""

    _current_phase: ClassVar[TestPhase]

    # Reasonable budget in number of trials where we expect the algo to match random search on the
    # Branin task.
    branin_task_max_trials: ClassVar[int] = 20

    max_trials: ClassVar[int]

    # The max number of trials required to sufficiently test out the last phase of the algorithm.
    # Used as a 'delta', so that max_trials is limited to the last phase n_trials + delta.
    _max_last_phase_trials: ClassVar[int] = 10

    # Fixtures available as class attributes:
    phase: ClassVar[pytest.fixture]  # type: ignore

    def __init_subclass__(cls) -> None:

        # Set the `algo_type` attribute, if necessary.
        if not hasattr(cls, "algo_type") or not cls.algo_type:
            if not cls.algo_name:
                raise RuntimeError(
                    f"Subclasses of BaseAlgoTests must set the algo_type or algo_name attributes, "
                    f"but class {cls.__qualname__} does not have either."
                )
            cls.algo_type = orion.algo.base.algo_factory.get_class(cls.algo_name)
        if not cls.algo_name:
            cls.algo_name = cls.algo_type.__name__.lower()

        # The first test phase should always have 0 as its n_trials, since algorithms are
        # supposed to work starting from 0 trials.
        assert cls.phases[0].n_trials == 0
        cls._current_phase = cls.phases[0]

        assert cls.phases == sorted(cls.phases, key=lambda v: v.n_trials)

        # Set a default value for the maximum number of trials programmatically.

        last_phase_start = cls.phases[-1].n_trials

        # NOTE: Because we auto-generate a max_trials for each class based on its phases, and we
        # have a default phase above, all subclasses of BaseAlgoTests will have an auto-generated
        # value for max_trials (even abstract ones for e.g. plugins).
        # For concrete test classes who use different phases than their parent, but don't define a
        # max_trials property, we want to auto-generate its value, and not use the max_trials of
        # their base class.
        # This is why we use `not in cls.__dict__` instead of `not hasattr(cls, "max_trials")`:
        if "max_trials" not in cls.__dict__:
            cls.max_trials = last_phase_start + cls._max_last_phase_trials
        elif last_phase_start > cls.max_trials - cls._max_last_phase_trials:
            raise ValueError(
                f"Test class {cls.__qualname__} isn't configured properly:\n"
                f"max_trials ({cls.max_trials}) should be larger than the start of the last phase "
                f"({cls.phases[-1].n_trials}) + delta ({cls._max_last_phase_trials}), for the last "
                f"phase to be properly tested. "
            )
        elif last_phase_start > cls.max_trials + cls._max_last_phase_trials:
            raise ValueError(
                f"Test class {cls.__qualname__} isn't configured properly:\n"
                f"max_trials ({cls.max_trials}) is larger than necessary, making tests longer to "
                f"run. Set max_trials to a value that is smaller than the start of the last phase "
                f"({last_phase_start}) + some delta (for example, {cls._max_last_phase_trials}), "
                f"so tests run efficiently."
            )

        # Inform the TestPhase object of their neighbours.
        # This can be used by tests to get the duration, start, end, etc of the test phases.
        previous: TestPhase | None = None
        for test_phase in cls.phases:
            if previous is not None:
                previous.next = test_phase
            test_phase.prev = previous
            previous = test_phase
        cls.phases[-1].next = cls.max_trials

        @pytest.fixture(
            name="phase",
            autouse=True,
            params=cls.phases,
            ids=[phase.name for phase in cls.phases],
        )
        @classmethod
        def phase(cls, request: pytest.FixtureRequest):
            """Fixture to parametrize tests with different phases."""
            test_phase: TestPhase = request.param  # type: ignore

            # Temporarily change the class attribute holding the current phase.
            original_phase = cls._current_phase
            cls._current_phase = test_phase
            # NOTE: If we want to actually use this spy stuff, We could create a spy for each
            # phase, and then in create_algo, after the force_observe, for each (phase, spy)
            # pair, check that the call_count is equal to phase.n_trials - prev_phase.n_trials
            # or something similar.
            yield test_phase
            cls._current_phase = original_phase

        # Store it somewhere on the class so it gets included in the test scope.
        cls.phase = phase  # type: ignore

    @pytest.fixture()
    def first_phase(self, phase: TestPhase):
        if phase != type(self).phases[0]:
            pytest.skip(reason="Test runs only on first phase.")
        return phase

    @pytest.fixture()
    def last_phase(self, phase: TestPhase):
        if phase != type(self).phases[-1]:
            pytest.skip(reason="Test runs only on last phase.")
        return phase

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
        n_observed_trials: int | None = None,
        **kwargs,
    ) -> SpaceTransformAlgoWrapper[AlgoType]:
        """Create the algorithm based on config.

        Also initializes the algorithm with the required number of random trials from the previous
        test phases before returning it.

        Parameters
        ----------
        config: dict, optional
            The configuration for the algorithm. ``cls.config`` will be used
            if ``config`` is ``None``.
        space: ``orion.algo.space.Space``, optional
            Space object to pass to algo. The output of ``cls.create_space()``
            will be used if ``space`` is ``None``.
        seed: int | Sequence[int], optional
            When passed, `seed_rng` is called before observing anything.
        n_observed_trials: int | None, optional
            Number of trials that the algorithm should have already observed when returned.
            When ``None`` (default), observes the number of trials at which the current phase
            begins.
            When set to 0, the algorithm will be freshly initialized.
        kwargs: dict
            Values to override algorithm configuration.
        """
        algo_kwargs = copy.deepcopy(config or cls.config)
        algo_kwargs.update(kwargs)

        original_space = space or cls.create_space()
        algo = create_algo(space=original_space, algo_type=cls.algo_type, **algo_kwargs)
        # TODO: Add a `max_trials` attribute on the BaseAlgorithm class.
        algo.algorithm.max_trials = cls.max_trials

        # Seed the randomness before we observe anything.
        if seed is not None:
            algo.seed_rng(seed)

        if n_observed_trials is None:
            n_observed_trials = cls._current_phase.n_trials

        if n_observed_trials:
            assert n_observed_trials > 0
            # Force the algo to observe the given number of trials.
            cls.force_observe(num=n_observed_trials, algo=algo)
        assert algo.n_observed == n_observed_trials
        return algo

    def update_space(self, test_space: dict) -> dict:
        """Get complete space configuration with partial overwrite

        The values passed in ``test_space`` will override the default values
        in ``self.config``.

        Parameters
        ----------
        test_space: dict
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
        cls, trials: list[Trial], algo: BaseAlgorithm, rng: numpy.random.RandomState
    ):
        """Make the algorithm observe trials

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
            Trials formatted as tuples of values
        algo: ``orion.algo.base.BaseAlgorithm``
            The algorithm used to observe trials.
        rng: ``numpy.random.RandomState``
            Random number generator to generate random objectives.
        """
        backward.algo_observe(
            algo,
            trials,
            [dict(objective=rng.normal()) for i in range(len(trials))],
        )

    @classmethod
    def get_num(cls, num: int):
        """Force number of trials to suggest

        Some algorithms must be tested with specific number of suggests at a time (ex: ASHA).
        This method can be overridden to change ``num`` based on the special needs.

        TODO: Remove this or give it a better name.
        """
        return num

    @classmethod
    def force_observe(cls, num: int, algo: BaseAlgorithm, seed: int = 1):
        """Force observe ``num`` trials.

        Parameters
        ----------
        num: int
            Number of trials to suggest and observe.
        algo: ``orion.algo.base.BaseAlgorithm``
            The algorithm that must suggest and observe.
        seed: int, optional
            The seed used to generate random objectives

        Raises
        ------
        RuntimeError
            - If the algorithm returns duplicates. Algorithms may return duplicates across workers,
              but in sequential scenarios as here, it should not happen.
            - If the algorithm fails to sample any trial at least 5 times.
        """
        rng = numpy.random.RandomState(seed)

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
            cls.observe_trials(trials, algo, rng)

        if failed >= MAX_FAILED:
            raise RuntimeError(
                f"Algorithm cannot sample more than {algo.n_observed} trials. Is it normal?"
            )

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
        assert len(trials) > 0
        assert trials[0] in space
        self.observe_trials(trials, algo, numpy.random.RandomState(1))

    @first_phase_only
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

    @pytest.mark.parametrize("seed", [123, 456])
    def test_seed_rng(self, seed: int):
        """Test that the seeding gives reproducible results."""
        numpy.random.seed(seed)
        algo = self.create_algo(seed=seed)

        trial_a = algo.suggest(1)[0]

        numpy.random.seed(seed)
        new_algo = self.create_algo(seed=seed)
        assert new_algo.n_observed == algo.n_observed
        trial_b = new_algo.suggest(1)[0]
        assert trial_b == trial_a

    @first_phase_only
    def test_seed_rng_init(self):
        """Test that if the algo has a `seed` constructor argument and a value is passed, the
        suggested trials are reproducible.
        """
        if "seed" not in inspect.signature(self.algo_type).parameters:
            pytest.skip(reason="algo does not have a seed as a constructor argument.")

        config = self.config.copy()
        config["seed"] = 1
        algo = self.create_algo(config=config)
        state = algo.state_dict

        first_trial = algo.suggest(1)[0]
        second_trial = algo.suggest(1)[0]
        assert first_trial != second_trial

        config = self.config.copy()
        config["seed"] = 2
        new_algo = self.create_algo(config=config)
        new_algo_state = new_algo.state_dict

        different_seed_trial = new_algo.suggest(1)[0]
        if _are_equal(new_algo_state, state):
            assert different_seed_trial == first_trial
        else:
            assert different_seed_trial != first_trial

        config = self.config.copy()
        config["seed"] = 1
        new_algo = self.create_algo(config=config)
        same_seed_trial = new_algo.suggest(1)[0]
        assert same_seed_trial == first_trial

    @pytest.mark.parametrize("seed", [123, 456])
    def test_state_dict(self, seed: int, phase: TestPhase):
        """Verify that resetting state makes sampling deterministic.

        The "source" algo is initialized at the start of each phase.
        The "target" algo instance is set to different initial conditions.
        This checks that it always gives the same suggestion as the original algo after set_state
        is used.
        """

        algo = self.create_algo(seed=seed)
        state = algo.state_dict
        a = algo.suggest(1)[0]

        # Create a new algo, without setting a seed.

        # The other algorithm is initialized at the start of the next phase.
        n_initial_trials = phase.end_n_trials
        # Use max_trials-1 so the algo can always sample at least one trial.
        if n_initial_trials == self.max_trials:
            n_initial_trials -= 1

        # NOTE: Seed is part of configuration, not state. Configuration is assumed to be the same
        #       for both algorithm instances.
        new_algo = self.create_algo(n_observed_trials=n_initial_trials, seed=seed)
        new_state = new_algo.state_dict
        b = new_algo.suggest(1)[0]
        # NOTE: For instance, if the algo doesn't have any RNG (e.g. GridSearch), this could be
        # True:
        if _are_equal(new_state, state):
            # If the state is the same, the trials should be the same.
            assert a == b
        else:
            # If the state is different, the trials should be different.
            assert a != b

        new_algo.set_state(state)
        c = new_algo.suggest(1)[0]
        assert a == c

    def test_suggest_n(self):
        """Verify that suggest returns correct number of trials if ``num`` is specified in ``suggest``."""
        algo = self.create_algo()
        trials = algo.suggest(5)
        assert trials is not None
        assert len(trials) == 5

    def test_has_suggested(self):
        """Verify that algorithm detects correctly if a trial was suggested"""
        algo = self.create_algo()
        a = algo.suggest(1)[0]
        assert algo.has_suggested(a)
        # NOTE: not algo.has_suggested(some random trial) is tested in test_has_suggested_statedict

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

    def test_observe(self):
        """Verify that algorithm observes trial without any issues"""
        algo = self.create_algo()

        a = algo.space.sample()[0]
        backward.algo_observe(algo, [a], [dict(objective=1)])

        b = algo.suggest(1)[0]
        backward.algo_observe(algo, [b], [dict(objective=2)])

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

    def test_n_suggested(self):
        """Verify that algorithm returns correct number of suggested trials"""
        algo = self.create_algo()
        initial = algo.n_suggested
        algo.suggest(1)
        assert algo.n_suggested == initial + 1

    def test_n_observed(self):
        """Verify that algorithm returns correct number of observed trials"""
        algo = self.create_algo()
        initial = algo.n_observed
        trials = algo.suggest(1)
        assert algo.n_observed == initial
        assert len(trials) == 1
        self.observe_trials(trials, algo, numpy.random.RandomState(1))
        assert algo.n_observed == initial + 1

    def test_real_data(self):
        """Test that algorithm supports real dimensions"""
        self.assert_dim_type_supported({"x": "uniform(0, 5)"})

    def test_int_data(self):
        """Test that algorithm supports integer dimensions"""
        self.assert_dim_type_supported({"x": "uniform(0, 5000, discrete=True)"})

    def test_cat_data(self):
        """Test that algorithm supports categorical dimensions"""
        self.assert_dim_type_supported(
            {  # Add 3 dims so that there exists many possible trials for the test
                "x": "choices(['a', 0.2, 1, None])",
                "y": "choices(['a', 0.2, 1, None])",
                "z": "choices(['a', 0.2, 1, None])",
            },
        )

    def test_logreal_data(self):
        """Test that algorithm supports logreal dimensions"""
        self.assert_dim_type_supported({"x": "loguniform(1, 5)"})

    def test_logint_data(self):
        """Test that algorithm supports loginteger dimensions"""
        self.assert_dim_type_supported({"x": "loguniform(1, 100, discrete=True)"})

    def test_shape_data(self):
        """Test that algorithm supports dimensions with shape"""
        self.assert_dim_type_supported({"x": "uniform(0, 5, shape=(3, 2))"})

    def test_broken_trials(self):
        """Test that algorithm can handle broken trials"""
        algo = self.create_algo()
        trial = algo.suggest(1)[0]
        trial.status = "broken"
        assert not algo.has_observed(trial)
        algo.observe([trial])
        assert algo.has_observed(trial)

    @first_phase_only
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
        # Prevent the algo from exiting early because of a max_trials limit.
        algo.algorithm.max_trials = None

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

    @last_phase_only
    def test_is_done_max_trials(self):
        """Test that algorithm will stop when max trials is reached"""
        algo = self.create_algo()
        # NOTE: Once https://github.com/Epistimio/orion/pull/883 is merged, we could update this to
        # force observe self.max_trials - phase.n_trials instead.
        self.force_observe(self.max_trials, algo)
        assert algo.is_done

    @first_phase_only
    def test_optimize_branin(self):
        """Test that algorithm optimizes a simple task comparably to random search."""
        max_trials = type(self).branin_task_max_trials
        task = Branin()
        space = self.create_space(task.get_search_space())
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = max_trials

        all_suggested_trials: list[Trial] = []
        all_objectives: list[float] = []

        # NOTE: Some algos work more effectively if they are asked to produce a batch of trials,
        # rather than a single trial at a time.
        max_batch_size = 5

        while len(all_suggested_trials) < max_trials and not algo.is_done:
            trials = algo.suggest(max_batch_size)
            all_suggested_trials.extend(trials)

            results = [task(**trial.params) for trial in trials]
            # NOTE: This is true for the branin task. If we ever test other tasks, this could vary.
            assert all(len(result) == 1 for result in results)
            new_objectives = [result[0]["value"] for result in results]
            all_objectives.extend(new_objectives)

            # NOTE: Not ideal that we have to unpack and repack the results of the task.
            results_for_backward_observe = [
                {"objective": objective} for objective in new_objectives
            ]
            backward.algo_observe(
                algo=algo, trials=trials, results=results_for_backward_observe
            )

        assert algo.is_done
        assert min(all_objectives) <= 10


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
