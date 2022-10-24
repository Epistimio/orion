"""Example usage and tests for :mod:`orion.algo.random`."""
from __future__ import annotations

import functools
import logging
from typing import ClassVar

import numpy as np
import pytest
from base import ExploitStub, ExploreStub, sample_trials
from pytest_mock import MockerFixture

from orion.algo.pbt.pbt import PBT, compute_fidelities
from orion.algo.space import Space
from orion.core.worker.primary_algo import SpaceTransform, create_algo
from orion.core.worker.transformer import ReshapedSpace, TransformedSpace
from orion.core.worker.trial import Trial
from orion.testing.algo import BaseAlgoTests, TestPhase

_create_algo = functools.partial(create_algo, PBT)


class TestComputeFidelities:
    def test_base_1(self):
        assert compute_fidelities(10, 10, 20, 1) == list(map(float, range(10, 21)))

    def test_other_bases(self):
        assert compute_fidelities(9, 2, 2**10, 2) == [2**i for i in range(1, 11)]

    @pytest.mark.xfail(reason="TODO: Test didn't have asserts, is now failing.")
    def test_fidelity_upgrades(self, space: Space):
        pbt = _create_algo(space).unwrap(PBT)
        fidelities = compute_fidelities(9, 2, 2**10, 2)
        assert pbt.fidelity_upgrades.keys() == fidelities[:-1]
        assert pbt.fidelity_upgrades.values() == fidelities[1:]


class TestPBTObserve:
    def test_triage_unknown_trial(self, space: Space):
        pbt = _create_algo(space).unwrap(PBT)
        trial = pbt.space.sample(1, seed=1)[0]
        trials_to_verify = pbt._triage([trial])

        assert trials_to_verify == []
        assert len(pbt.lineages) == 0

    @pytest.mark.parametrize("status", ["new", "reserved", "interrupted"])
    def test_triage_root_not_ready(self, status: str, space: Space):
        pbt = _create_algo(space).unwrap(PBT)

        trial = sample_trials(pbt.space, num=1, status=status)[0]

        pbt.register(trial)

        trials_to_verify = pbt._triage([trial])

        assert trials_to_verify == []
        assert pbt.has_suggested(trial)
        assert not pbt.has_observed(trial)
        assert len(pbt.lineages) == 1

    @pytest.mark.parametrize("status", ["broken", "completed"])
    def test_triage_root_ready(self, status: str, space: Space):
        pbt = _create_algo(space).unwrap(PBT)

        trial = sample_trials(pbt.space, num=1, status="new")[0]

        pbt.register(trial)

        trial.status = status
        trial._results.append(Trial.Result(name="objective", type="objective", value=1))

        trials_to_verify = pbt._triage([trial])

        assert trials_to_verify == [trial]

        assert pbt.has_suggested(trial)
        assert pbt.has_observed(trial)
        assert len(pbt.lineages) == 1

    @pytest.mark.parametrize("status", ["broken", "completed"])
    def test_triage_root_observed(self, status: str, space: Space):
        pbt = _create_algo(space).unwrap(PBT)

        trial = sample_trials(pbt.space, num=1, status="completed", objective=1)[0]

        pbt.register(trial)

        trials_to_verify = pbt._triage([trial])

        assert trials_to_verify == []

        assert pbt.has_suggested(trial)
        assert pbt.has_observed(trial)
        assert len(pbt.lineages) == 1

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_dont_queue_broken_root_for_promotions(self, space: Space):
        pbt = _create_algo(space).unwrap(PBT)

        trial = sample_trials(pbt.space, num=1, status="broken")[0]
        pbt.register(trial)

        # Should not queue anything
        pbt._queue_trials_for_promotions([trial])
        assert len(pbt._queue) == 0

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_queue_broken_trials_for_promotions(self, space: Space):
        pbt = _create_algo(space).unwrap(PBT)
        trial = sample_trials(pbt.space, num=1, status="completed", objective=1)[0]
        pbt.register(trial)

        new_trial = trial.branch(params={"f": pbt.fidelities[trial.params["f"]]})
        pbt.lineages.fork(trial, new_trial)

        new_trial.status = "broken"
        pbt.register(new_trial)

        # Should queue the parent of the broken trial
        pbt._queue_trials_for_promotions([new_trial])
        assert len(pbt._queue) == 1
        assert pbt._queue[0].id == trial.id

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_queue_broken_trials_from_jump_for_promotions(self, space: Space):
        pbt = _create_algo(space).unwrap(PBT)

        parent_trial = sample_trials(pbt.space, num=1, status="completed", objective=1)[
            0
        ]
        base_trial = sample_trials(
            pbt.space, num=1, seed=2, status="completed", objective=1
        )[0]

        pbt.register(parent_trial)
        pbt.register(base_trial)

        new_trial = parent_trial.branch(
            params={"f": pbt.fidelities[parent_trial.params["f"]]}
        )
        pbt.lineages.fork(parent_trial, new_trial)
        pbt.lineages.set_jump(base_trial, new_trial)

        new_trial.status = "broken"
        pbt.register(new_trial)

        # Should queue the parent of the broken trial
        pbt._queue_trials_for_promotions([new_trial])
        assert len(pbt._queue) == 1
        assert pbt._queue[0].id == base_trial.id

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_queue_completed_trials_for_promotions(self, space: Space):
        pbt = _create_algo(space).unwrap(PBT)

        trial = sample_trials(pbt.space, num=1, status="completed", objective=1)[0]
        pbt.register(trial)

        # Should queue the trial itself
        pbt._queue_trials_for_promotions([trial])
        assert len(pbt._queue) == 1
        assert pbt._queue[0].id == trial.id

        new_trial = trial.branch(params={"f": pbt.fidelities[trial.params["f"]]})
        pbt.lineages.fork(trial, new_trial)

        new_trial.status = "completed"
        new_trial._results.append(
            Trial.Result(name="objective", type="objective", value=1)
        )
        pbt.register(new_trial)

        # Should queue the parent of the broken trial
        pbt._queue_trials_for_promotions([new_trial])
        assert len(pbt._queue) == 2
        assert pbt._queue[1].id == new_trial.id

    @pytest.mark.parametrize("status", ["new", "reserved", "interrupted"])
    def test_dont_queue_pending_trials_for_promotions(self, space: Space, status: str):
        pbt = _create_algo(space).unwrap(PBT)

        trial = sample_trials(pbt.space, num=1, status=status)[0]
        pbt.register(trial)

        # Should not queue anything
        pbt._queue_trials_for_promotions([trial])
        assert len(pbt._queue) == 0


class TestPBTSuggest:
    def test_generate_offspring_unknown_trial(self, space: Space):

        pbt = _create_algo(space).unwrap(PBT)
        trial = sample_trials(pbt.space, 1)[0]
        with pytest.raises(RuntimeError, match="Trying to fork a trial that"):
            pbt._generate_offspring(trial)

    def test_generate_offspring_exploit_skip(self, space: Space):

        pbt = _create_algo(
            space,
            exploit=ExploitStub(skip=True).configuration,
        ).unwrap(PBT)
        trial = sample_trials(pbt.space, 1, status="completed", objective=1)[0]
        pbt.register(trial)

        trial_to_branch, new_trial = pbt._generate_offspring(trial)
        assert trial_to_branch is None
        assert new_trial is None

    def test_generate_offspring_exploit_promote(self, space: Space):

        pbt = _create_algo(
            space,
            exploit=ExploitStub().configuration,
            explore=ExploreStub(no_call=True).configuration,
        ).unwrap(PBT)
        trial = sample_trials(pbt.space, 1, status="completed", objective=1)[0]

        # Apply the transformation and revert it to have lossy effect (like small precision)
        assert isinstance(pbt.space, (TransformedSpace, ReshapedSpace))
        trial = pbt.space.transform(pbt.space.reverse(pbt.space.transform(trial)))

        pbt.register(trial)

        new_params_expected = trial.params
        new_params_expected["f"] = 10.9

        trial_to_branch, new_trial = pbt._generate_offspring(trial)
        assert trial_to_branch is trial
        assert new_trial is not None
        assert new_trial.params == new_params_expected

    def test_generate_offspring_exploit_branch(self, space: Space):
        pbt = _create_algo(
            space,
            exploit=ExploitStub(rval="toset").configuration,
            explore=ExploreStub(rval="toset").configuration,
        ).unwrap(PBT)

        trials = sample_trials(pbt.space, 3, status="completed", objective=1)

        trial_to_promote = trials[0]
        exploited_trial = trials[1]
        new_params_expected = trials[2].params

        pbt.exploit_func.rval = exploited_trial
        pbt.explore_func.rval = new_params_expected

        # Make sure they are different
        assert new_params_expected != trial_to_promote.params
        assert new_params_expected != exploited_trial.params

        pbt.register(trials[0])
        pbt.register(trials[1])

        trial_to_branch, new_trial = pbt._generate_offspring(trial_to_promote)

        new_params_expected["f"] = 10.9

        assert new_trial.experiment == trial_to_branch.experiment
        assert trial_to_branch is exploited_trial
        assert new_trial is not None
        assert new_trial.params["f"] == new_params_expected["f"]
        assert new_trial.params == new_params_expected

    def test_generate_offspring_timeout(
        self, space: Space, caplog: pytest.LogCaptureFixture
    ):

        pbt = _create_algo(
            space,
            exploit=ExploitStub(rval=None).configuration,
            explore=ExploreStub(rval="toset").configuration,
            fork_timeout=0.05,
        ).unwrap(PBT)
        trial = sample_trials(pbt.space, 1, status="completed", objective=1)[0]
        pbt.explore_func.rval = trial.params

        pbt.register(trial)
        parent = trial.branch(params={"f": pbt.fidelities[space["f"].low]})
        pbt.register(parent)

        with caplog.at_level(logging.INFO):
            trial_to_branch, new_trial = pbt._generate_offspring(trial)

        assert "Could not generate unique new parameters" in caplog.records[-1].message
        assert trial_to_branch is None
        assert new_trial is None

    def test_generate_offspring_retry_using_same_trial(
        self,
        space: Space,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that when exploit returns another trial, the base one is reused and case of
        duplicate samples
        """
        pbt = _create_algo(
            space,
            exploit=ExploitStub(rval="toset", should_receive="toset").configuration,
            explore=ExploreStub(rval="toset").configuration,
            fork_timeout=0.0001,
        ).unwrap(PBT)

        trials = sample_trials(pbt.space, 3, status="completed", objective=1)
        parent_trial = trials[0]
        base_trial = trials[1]
        sample_params = trials[2].params

        pbt.exploit_func.rval = parent_trial
        pbt.exploit_func.should_receive = base_trial
        pbt.explore_func.rval = sample_params

        pbt.register(parent_trial)
        pbt.register(base_trial)

        # The trial sampled will already be registered
        sample_params["f"] = pbt.fidelities[space["f"].low]
        child = parent_trial.branch(params=sample_params)
        pbt.register(child)

        # Exploit will return parent_trial, but Explore will return params of child, sampling
        # a duplite, since child is already registered. ExploitStub.should_receive will
        # test that base_trial is passed as expected to exploit when attempting more attempts
        # of exploit and explore.
        with caplog.at_level(logging.INFO):
            trial_to_branch, new_trial = pbt._generate_offspring(base_trial)

        assert "Could not generate unique new parameters" in caplog.records[-1].message
        assert trial_to_branch is None
        assert new_trial is None

    def test_fork_lineages_empty_queue(self, space: Space):
        pbt = _create_algo(space).unwrap(PBT)
        assert pbt._fork_lineages(10) == []

    def test_fork_lineages_skip_and_requeue_trials(self, space: Space):
        num = 10
        pbt = _create_algo(
            space,
            exploit=ExploitStub(skip=True).configuration,
        ).unwrap(PBT)

        trials = sample_trials(pbt.space, num, status="completed", objective=1)

        for trial in trials:
            pbt.register(trial)

        pbt._queue = trials[:]

        assert pbt._fork_lineages(num) == []
        assert len(pbt._queue) == num
        assert pbt._queue == trials

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_fork_lineages_promote_trial(self, space: Space):
        num = 10
        pbt = _create_algo(
            space,
            exploit=ExploitStub(skip=None).configuration,
        ).unwrap(PBT)

        trials = sample_trials(pbt.space, num, status="completed", objective=1)

        for trial in trials:
            pbt.register(trial)

        pbt._queue = trials[:]

        branched_trials = pbt._fork_lineages(num)
        assert len(trials) == num
        assert len(branched_trials) == num
        assert pbt._queue == []

        for trial, branched_trial in zip(trials, branched_trials):
            expected_params = trial.params
            expected_params["f"] = 10.9
            assert branched_trial.params == expected_params

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_fork_lineages_branch_trials(self, space: Space):
        num = 10

        pbt = _create_algo(
            space,
            exploit=ExploitStub(rval="toset").configuration,
            fork_timeout=0.05,
        ).unwrap(PBT)

        trials = sample_trials(pbt.space, num + 1, status="completed", objective=1)
        trial_to_branch = trials[-1]
        pbt.exploit_func.rval = trial_to_branch
        for trial in trials:
            pbt.register(trial)

        pbt._queue = trials[:-1]

        branched_trials = pbt._fork_lineages(num)

        assert len(trials) == num + 1
        assert len(branched_trials) == num
        assert pbt._queue == []

        for trial, branched_trial in zip(trials, branched_trials):
            # Check if parent is correct
            assert branched_trial.parent == trial_to_branch.id
            # Check in lineage if jump is set from correct base trial
            assert pbt.lineages.get_lineage(branched_trial).base.item.id == trial.id
            # Check if params are not duplicated
            should_not_be_params = trial_to_branch.params
            should_not_be_params["f"] = 10.9
            assert branched_trial.params["f"] == should_not_be_params["f"]
            assert branched_trial.params != should_not_be_params

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_fork_lineages_branch_duplicates(
        self, space: Space, caplog: pytest.LogCaptureFixture
    ):
        num = 10
        pbt = _create_algo(
            space,
            exploit=ExploitStub(rval="toset").configuration,
            explore=ExploreStub(rval="toset").configuration,
            fork_timeout=0.05,
        ).unwrap(PBT)

        trials = sample_trials(pbt.space, num + 1, status="completed", objective=1)
        new_params_expected = trials[-1].params
        pbt.exploit_func.rval = trials[-1]
        pbt.explore_func.rval = new_params_expected
        for trial in trials:
            pbt.register(trial)

        pbt._queue = trials[:-1]

        with caplog.at_level(logging.INFO):
            branched_trials = pbt._fork_lineages(num)

        assert "Could not generate unique new parameters" in caplog.records[-1].message
        assert len(branched_trials) == 1
        assert branched_trials[0].params == {**trials[-1].params, **{"f": 10.9}}

        # First queue.pop is fine, fails on second queue.pop, trial is reinserted at beginning
        assert len(pbt._queue) == num - 1
        assert (
            len(trials) == num + 1
        )  # make sure trials list was not modified during execution.
        assert pbt._queue[0].params == trials[1].params

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_fork_lineages_num_larger_than_queue(self, space: Space):
        num = 10
        pbt = _create_algo(
            space,
            exploit=ExploitStub(rval=None).configuration,
        ).unwrap(PBT)

        trials = sample_trials(pbt.space, num, status="completed", objective=1)
        for trial in trials:
            pbt.register(trial)

        pbt._queue = trials[:]

        num_fork = 4
        branched_trials = pbt._fork_lineages(num_fork)

        assert len(branched_trials) == num_fork
        assert len(pbt._queue) == num - num_fork

        trial_ids = [trial.id for trial in trials]

        assert [trial.parent for trial in branched_trials] == trial_ids[:num_fork]
        assert [trial.id for trial in pbt._queue] == trial_ids[num_fork:]

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_fork_lineages_num_smaller_than_queue(self, space: Space):
        num = 4
        pbt = _create_algo(
            space,
            exploit=ExploitStub(rval=None).configuration,
        ).unwrap(PBT)

        trials = sample_trials(pbt.space, num, status="completed", objective=1)
        for trial in trials:
            pbt.register(trial)

        pbt._queue = trials[:]

        num_fork = 10
        branched_trials = pbt._fork_lineages(num_fork)

        assert len(branched_trials) == num
        assert len(pbt._queue) == 0

        trial_ids = [trial.id for trial in trials]

        assert [trial.parent for trial in branched_trials] == trial_ids

    def test_suggest_num_population_size_sample(
        self, space: Space, mocker: MockerFixture
    ):
        population_size = 10
        pbt = _create_algo(space, population_size=population_size).unwrap(PBT)

        pbt_sample_mock = mocker.spy(pbt, "_sample")
        pbt_fork_mock = mocker.spy(pbt, "_fork_lineages")

        num = 6
        assert len(pbt.suggest(num)) == num

        pbt_sample_mock.assert_called_with(num)
        pbt_fork_mock.assert_called_with(0)

        assert len(pbt.suggest(num)) == 4

        pbt_sample_mock.assert_called_with(4)
        pbt_fork_mock.assert_called_with(2)

    def test_suggest_num_population_size_sample_broken(
        self, space: Space, mocker: MockerFixture
    ):
        population_size = 10
        pbt = _create_algo(space, population_size=population_size).unwrap(PBT)

        pbt_sample_mock = mocker.spy(pbt, "_sample")
        pbt_fork_mock = mocker.spy(pbt, "_fork_lineages")

        num = 10
        trials = pbt.suggest(num)
        assert len(trials) == num

        pbt_sample_mock.assert_called_with(num)
        pbt_fork_mock.assert_called_with(0)

        n_broken = 3
        for trial in trials[:n_broken]:
            trial.status = "broken"

        pbt.observe(trials)

        assert len(pbt.suggest(num)) == n_broken

        # 3 trials are broken, need to resample 3 trials, and can try to fork 7 trials
        pbt_sample_mock.assert_called_with(n_broken)
        pbt_fork_mock.assert_called_with(7)

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_suggest_num_population_size_fork_completed(
        self, space: Space, mocker: MockerFixture
    ):
        population_size = 10
        pbt = _create_algo(
            space,
            population_size=population_size,
            exploit=ExploitStub(rval=None).configuration,
        ).unwrap(PBT)

        pbt_sample_mock = mocker.spy(pbt, "_sample")
        pbt_fork_mock = mocker.spy(pbt, "_fork_lineages")

        num = 4
        trials = pbt.suggest(num)
        assert len(trials) == num

        pbt_sample_mock.assert_called_with(num)
        pbt_fork_mock.assert_called_with(0)

        n_completed = 3
        for trial in trials[:n_completed]:
            trial.exp_working_dir = "/nothing"
            trial.status = "completed"
            trial._results.append(
                Trial.Result(name="objective", type="objective", value=1)
            )

        pbt.observe(trials)
        assert len(pbt._queue) == n_completed

        # There are 4 trials sampled, out of which 3 are completed. Still missing 6 trials
        # for base population.
        assert len(pbt.suggest(num)) == num
        pbt_sample_mock.assert_called_with(num)
        pbt_fork_mock.assert_called_with(0)

        # There are 8 trials sampled, out of which 3 are completed. Still missing 2 trials
        # for base population.
        assert len(pbt.suggest(num)) == num
        pbt_sample_mock.assert_called_with(2)
        pbt_fork_mock.assert_called_with(2)


population_size = 10
generations = 5


@pytest.mark.usefixtures("no_shutil_copytree")
class TestGenericPBT(BaseAlgoTests):
    algo_name = "pbt"
    algo_type = PBT
    max_trials = population_size * generations
    config = {
        "seed": 123456,
        "population_size": population_size,
        "generations": generations,
        "exploit": {
            "of_type": "PipelineExploit",
            "exploit_configs": [
                {
                    "of_type": "BacktrackExploit",
                    "min_forking_population": population_size / 2,
                    "candidate_pool_ratio": 0.0,
                    "truncation_quantile": 1.0,
                },
                {
                    "of_type": "TruncateExploit",
                    "min_forking_population": population_size / 2,
                    "candidate_pool_ratio": 0.3,
                    "truncation_quantile": 0.9,
                },
            ],
        },
        "explore": {
            "of_type": "PipelineExplore",
            "explore_configs": [
                {
                    "of_type": "ResampleExplore",
                    "probability": 0.3,
                },
                {
                    "of_type": "PerturbExplore",
                    "factor": 1.5,
                    "volatility": 0.005,
                },
            ],
        },
        "fork_timeout": 5,
    }
    space = {"x": "uniform(0, 1)", "y": "uniform(0, 1)", "f": "fidelity(1, 10, base=1)"}

    phases: ClassVar[list[TestPhase]] = [
        TestPhase("random", 0, "space.sample"),
        TestPhase("generation_2", 2 * population_size, "_generate_offspring"),
        TestPhase("generation_3", 3 * population_size, "_generate_offspring"),
    ]

    def test_no_fidelity(self):
        space = self.create_space({"x": "uniform(0, 1)", "y": "uniform(0, 1)"})

        with pytest.raises(
            RuntimeError, match="PBT cannot be used if space does not contain"
        ):
            self.create_algo(space=space)

    @pytest.mark.skip(
        reason="There are no good reasons to use PBT if search space is so small"
    )
    def test_is_done_cardinality(self):
        pass

    @pytest.mark.parametrize("num", [100000, 1])
    def test_is_done_max_trials(self, num: int):
        space = self.create_space()

        MAX_TRIALS = 10
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = MAX_TRIALS

        rng = np.random.RandomState(123456)

        objective = 0
        while not algo.is_done:
            trials = algo.suggest(num)
            assert trials is not None
            if trials:
                self.observe_trials(trials, algo, rng)

        # BPT should ignore max trials.
        assert algo.n_observed > MAX_TRIALS
        # BPT should stop when all trials of last generation are completed.
        assert algo.n_observed == population_size * (generations + 1)
        assert algo.is_done

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        pass

    def assert_callbacks(self, spy, num: int, algo: SpaceTransform[PBT]):
        def check_population_size(gen_population_size, depth, expected):
            assert (
                gen_population_size == expected
            ), f"population of {gen_population_size} at depth {depth}, should be {expected}"

        pbt = algo.algorithm
        remaining_num = num

        for depth in range(generations):
            gen_population_size = len(pbt.lineages.get_trials_at_depth(depth))
            if remaining_num > population_size:
                expected_population_size = population_size
            else:
                expected_population_size = remaining_num

            check_population_size(gen_population_size, depth, expected_population_size)

            remaining_num = max(remaining_num - expected_population_size, 0)
