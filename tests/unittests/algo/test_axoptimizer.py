"""Perform integration tests for `orion.algo.ax`."""
import statistics as stats

import pytest

from orion.algo.axoptimizer import has_Ax
from orion.benchmark.task.base import BenchmarkTask
from orion.core.utils import backward
from orion.testing.algo import BaseAlgoTests, TestPhase, phase

if not has_Ax:
    pytest.skip("skipping Ax tests", allow_module_level=True)
else:
    import numpy
    from botorch.test_functions.multi_objective import BraninCurrin
    from torch import Tensor

N_INIT = 5
TOL = 0.1


class BraninCurrinTask(BenchmarkTask):
    r"""Two objective problem composed of the Branin and Currin functions.

    Branin (rescaled):

        f(x) = (
        15*x_1 - 5.1 * (15 * x_0 - 5) ** 2 / (4 * pi ** 2) + 5 * (15 * x_0 - 5)
        / pi - 5
        ) ** 2 + (10 - 10 / (8 * pi)) * cos(15 * x_0 - 5))

    Currin:

        f(x) = (1 - exp(-1 / (2 * x_1))) * (
        2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
        ) / 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20

    """

    def __init__(self, max_trials=20):
        self._branincurrin = BraninCurrin()
        super().__init__(max_trials=max_trials)

    # pylint: disable=arguments-differ
    def call(self, x):
        """Evaluate 2-D branin and currin functions."""
        _y = self._branincurrin(Tensor(x))
        return [
            dict(name="branin", type="objective", value=_y[0].item()),
            dict(name="currin", type="statistic", value=_y[1].item()),
        ]

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {"x": "uniform(0, 1, shape=2, precision=10)"}

        return rspace


class TestAxOptimizer(BaseAlgoTests):
    """Test suite for algorithm AxOptimizer"""

    algo_name = "axoptimizer"
    max_trials = 20
    config = {
        "seed": 1234,  # Because this is so random
        "n_initial_trials": N_INIT,
        "extra_objectives": set(),
        "constraints": [],
    }
    phases: ClassVar[list[TestPhase]] = [
        TestPhase("Sobol", 0, "space.sample"),
        TestPhase("BO", N_INIT, "space.sample"),
    ]

    def test_configuration_fail(self):
        """Test that Ax configuration is valid"""
        with pytest.raises(AssertionError) as exc_info:
            self.create_algo(
                config={**self.config, "constraints": ["constraint = 3"]},
            )
        assert exc_info.value
        with pytest.raises(AssertionError) as exc_info:
            self.create_algo(
                config={**self.config, "constraints": ["constraint < 10"]},
            )
        assert exc_info.value

    def test_is_done_cardinality(self, *args, **kwargs):
        # Set higher max_trials to explore all cardinality space
        _max_trials = self.max_trials
        self.max_trials = 200
        super().test_is_done_cardinality(*args, **kwargs)
        self.max_trials = _max_trials

    def test_seed_rng(self, mocker, num, attr):
        """Test that the seeding gives reproducibile results."""
        algo = self.create_algo()

        seed = numpy.random.randint(10000)
        algo.seed_rng(seed)
        spy = self.spy_phase(mocker, num, algo, attr)
        trials = algo.suggest(1)
        trials[0].id != algo.suggest(1)[0].id

        new_algo = self.create_algo()
        new_algo.seed_rng(seed)
        self.force_observe(algo.n_observed, new_algo)
        numpy.testing.assert_allclose(
            numpy.array(list(trials[0].params.values())).astype(float),
            numpy.array(list(new_algo.suggest(1)[0].params.values())).astype(float),
            atol=TOL,
            rtol=TOL,
        )

        self.assert_callbacks(spy, num + 1, new_algo)

    def test_seed_rng_init(self, mocker, num, attr):
        """Test that the seeding gives reproducibile results."""
        algo = self.create_algo(seed=1)

        spy = self.spy_phase(mocker, num, algo, attr)
        trials = algo.suggest(1)
        algo.suggest(1)[0].id != trials[0].id

        new_algo = self.create_algo(seed=2)
        self.force_observe(algo.n_observed, new_algo)
        assert new_algo.suggest(1)[0].id != trials[0].id

        new_algo = self.create_algo(seed=1)
        self.force_observe(algo.n_observed, new_algo)
        numpy.testing.assert_allclose(
            numpy.array(list(trials[0].params.values())).astype(float),
            numpy.array(list(new_algo.suggest(1)[0].params.values())).astype(float),
            atol=TOL,
            rtol=TOL,
        )

        self.assert_callbacks(spy, num + 1, new_algo)

    @phase
    def test_state_dict(self, mocker, num, attr):
        """Verify that resetting state makes sampling deterministic"""
        algo = self.create_algo()

        seed = numpy.random.randint(10000)
        algo.seed_rng(seed)
        spy = self.spy_phase(mocker, max(num, 1), algo, attr)
        state = algo.state_dict
        trial = algo.suggest(1)[0]

        algo.set_state(state)
        numpy.testing.assert_allclose(
            numpy.array(list(trial.params.values())).astype(float),
            numpy.array(list(algo.suggest(1)[0].params.values())).astype(float),
            atol=TOL,
            rtol=TOL,
        )

        new_algo = self.create_algo()
        assert trial.id != new_algo.suggest(1)[0].id

        new_algo.set_state(state)
        numpy.testing.assert_allclose(
            numpy.array(list(trial.params.values())).astype(float),
            numpy.array(list(new_algo.suggest(1)[0].params.values())).astype(float),
            atol=TOL,
            rtol=TOL,
        )

        self.assert_callbacks(spy, num + 1, algo)

    def test_optimize_multi_objectives(self):
        """Test that algorithm optimizes somehow (this is on-par with random search)"""
        _max_trials = 20
        task = BraninCurrinTask()
        space = self.create_space(task.get_search_space())
        algo = self.create_algo(
            config={**self.config, "extra_objectives": ["statistic"]}, space=space
        )
        algo.algorithm.max_trials = _max_trials
        safe_guard = 0
        trials = []
        objectives = []
        while trials or not algo.is_done:
            if safe_guard >= _max_trials:
                break

            if not trials:
                trials = algo.suggest(_max_trials - len(objectives))

            trial = trials.pop(0)
            results = task(trial.params["x"])
            objectives.append((results[0]["value"], results[1]["value"]))
            backward.algo_observe(
                algo,
                [trial],
                [dict(objective=objectives[-1][0], statistic=objectives[-1][1])],
            )
            safe_guard += 1

        rand_objectives = []
        for trial in algo.space.sample(len(objectives)):
            results = task(trial.params["x"])
            rand_objectives.append((results[0]["value"], results[1]["value"]))

        objectives_branin, objectives_currin = list(zip(*objectives))
        _, rand_objectives_currin = list(zip(*rand_objectives))

        assert algo.is_done
        # branin
        assert min(objectives_branin) <= 10
        # currin
        assert min(objectives_currin) <= min(rand_objectives_currin)

    def test_objectives_constraints(self):
        """Test that algorithm optimizes somehow (this is on-par with random search)"""
        _max_trials = 20
        task = BraninCurrinTask()
        space = self.create_space(task.get_search_space())
        algo = self.create_algo(
            config={
                **self.config,
                "constraints": ["constraint >= 3", "constraint <= 10"],
            },
            space=space,
        )
        algo.algorithm.max_trials = _max_trials
        safe_guard = 0
        trials = []
        objectives = []
        while trials or not algo.is_done:
            if safe_guard >= _max_trials:
                break

            if not trials:
                trials = algo.suggest(_max_trials - len(objectives))

            trial = trials.pop(0)
            results = task(trial.params["x"])
            objectives.append((results[0]["value"], results[1]["value"]))
            backward.algo_observe(
                algo,
                [trial],
                [dict(objective=objectives[-1][0], constraint=objectives[-1][1])],
            )
            safe_guard += 1

        objectives_branin, objectives_currin = list(zip(*objectives))

        assert algo.is_done
        # branin
        assert (
            min(objectives_branin)
            <= stats.mean(objectives_branin) - stats.stdev(objectives_branin) * 0.7
        )
        # currin
        assert 3 <= stats.mean(objectives_currin[-5:]) <= 10
