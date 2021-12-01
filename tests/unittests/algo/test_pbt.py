# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.random`."""
import shutil

import numpy
import pytest

from orion.algo.pbt import (
    compute_fidelities,
    perturb,
    perturb_cat,
    perturb_int,
    perturb_real,
    resample,
    resample_or_perturb,
    truncate,
    Lineage,
)
from orion.algo.space import Integer, Real, Space
from orion.core.io.space_builder import SpaceBuilder
from orion.testing.algo import BaseAlgoTests
from orion.core.utils.pptree import print_tree


def build_full_tree(depth, child_per_parent=2, starting_objective=1):
    """Build a full tree

    Parameters
    ----------
    depth: int
        Depth of the tree

    child_per_parent: int, optional
        Number of child per node. Default: 2
    """

    def create_node_item(node_index):
        return TrialStub(id=f"id-{node_index}", objective=node_index)

    node_index = starting_objective
    root = Lineage(create_node_item(node_index))
    node_index += 1
    node_buffer = [root]
    next_nodes = []
    for i in range(depth - 1):
        for node in node_buffer:
            for k in range(child_per_parent):
                next_nodes.append(Lineage(create_node_item(node_index), parent=node))
                node_index += 1
        node_buffer = next_nodes
        next_nodes = []

    print_tree(root, nameattr="tree_name")

    return root


class RNGStub:
    pass


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
def trials(tmp_path, space):
    trials = space.sample(100, seed=1)
    for i, trial in enumerate(trials):
        trial.exp_working_dir = tmp_path
        trial.status = "completed"
        trial._results.append(trial.Result(name="objective", type="objective", value=i))

    return trials


class TestComputeFidelities:
    def test_base_1(self):
        assert compute_fidelities(10, 10, 20, 1).tolist() == list(
            map(float, range(10, 21))
        )

    def test_other_bases(self):
        assert compute_fidelities(9, 2, 2 ** 10, 2).tolist() == [
            2 ** i for i in range(1, 11)
        ]


class ObjectiveStub:
    def __init__(self, value):
        self.value = value


class TrialStub:
    def __init__(self, working_dir="/some_path", objective=None, id=None):
        self.id = id
        self.working_dir = working_dir
        if objective:
            self.objective = ObjectiveStub(objective)
        else:
            self.objective = None

    def __repr__(self):
        return self.id


class TestLineage:
    def test_register(self):
        item = [0]
        lineage = Lineage(item)
        assert lineage.item == item
        assert lineage.item is not item

        item = [1]
        lineage.register(item)
        assert lineage.item == item
        assert lineage.item is not item

    def test_fork(self, mocker):
        path = "/some_path"
        trial = TrialStub(path)
        lineage = Lineage(trial)

        new_path = "/another_path"
        new_trial = TrialStub(new_path)

        mocker.patch("shutil.copytree")
        new_lineage = lineage.fork(new_trial)
        shutil.copytree.assert_called_once_with(path, new_path)

        assert new_lineage.item.working_dir == new_trial.working_dir
        assert new_lineage.parent is lineage
        assert lineage.children[0] is new_lineage

    def test_fork_identical_new_trial(self):
        lineage = Lineage(TrialStub(id="my-id"))
        with pytest.raises(
            RuntimeError, match="The new trial new-id has the same working directory"
        ):
            lineage.fork(TrialStub(id="new-id"))

        assert lineage.children == []

    def test_set_jump(self):
        parent_lineage = Lineage(1)
        child_lineage = Lineage(2)
        parent_lineage.set_jump(child_lineage)

        assert child_lineage.parent is None
        assert child_lineage.jumps == []
        assert child_lineage.base is parent_lineage

        assert parent_lineage.children == []
        assert parent_lineage.jumps == [child_lineage]
        assert parent_lineage.base is None

    def test_set_jump_twice(self):
        parent_lineage = Lineage(1)
        child_lineage = Lineage(2)
        parent_lineage.set_jump(child_lineage)

        another_child_lineage = Lineage(3)
        parent_lineage.set_jump(another_child_lineage)

        assert child_lineage.parent is None
        assert child_lineage.jumps == []
        assert child_lineage.base is parent_lineage

        assert another_child_lineage.parent is None
        assert another_child_lineage.jumps == []
        assert another_child_lineage.base is parent_lineage

        assert parent_lineage.children == []
        assert parent_lineage.jumps == [child_lineage, another_child_lineage]
        assert parent_lineage.base is None

    def test_set_jump_to_old_node(self):
        parent_lineage = Lineage(1)
        child_lineage = Lineage(2)
        parent_lineage.set_jump(child_lineage)

        another_child_lineage = Lineage(3)

        with pytest.raises(RuntimeError, match="Trying to jump to an existing node"):
            another_child_lineage.set_jump(child_lineage)

        assert child_lineage.parent is None
        assert child_lineage.jumps == []
        assert child_lineage.base is parent_lineage

        assert another_child_lineage.parent is None
        assert another_child_lineage.jumps == []
        assert another_child_lineage.base is None

        assert parent_lineage.children == []
        assert parent_lineage.jumps == [child_lineage]
        assert parent_lineage.base is None

    def test_get_best_trial_empty(self):
        trial = TrialStub(id="id-1", objective=1)
        lineage = Lineage(trial)
        assert lineage.get_best_trial().id == "id-1"

    def test_get_best_trial_straigth_lineage(self):
        root = build_full_tree(4)
        leafs = root.get_nodes_at_depth(3)
        assert leafs[0].item.id == "id-8"
        assert leafs[0].get_best_trial() == root.item
        assert leafs[1].get_best_trial() == root.item
        leafs[0].item.objective.value = -1
        # Now best trial is leaf on first branch
        assert leafs[0].get_best_trial() == leafs[0].item
        # But still root for second branch
        assert leafs[1].get_best_trial() == root.item

        third_row = root.get_nodes_at_depth(2)
        assert third_row[0].item.id == "id-4"
        assert third_row[0].get_best_trial() == root.item
        assert third_row[1].get_best_trial() == root.item

        third_row[0].item.objective.value = -2
        # Now best trial is third node on first branch
        assert third_row[0].get_best_trial() == third_row[0].item
        # But still root for second branch
        assert third_row[1].get_best_trial() == root.item
        # And third node on full first and second branches
        assert leafs[0].get_best_trial() == third_row[0].item
        assert leafs[1].get_best_trial() == third_row[0].item
        # But not for third branch
        assert leafs[2].get_best_trial() == root.item

        second_row = root.get_nodes_at_depth(1)
        assert second_row[0].item.id == "id-2"
        assert second_row[0].get_best_trial() == root.item
        assert second_row[1].get_best_trial() == root.item

        second_row[0].item.objective.value = -3
        # Now best trial is second node on first branch
        assert second_row[0].get_best_trial() == second_row[0].item
        # But still root for second branch
        assert second_row[1].get_best_trial() == root.item
        # And second node on full 4 first branches
        assert leafs[0].get_best_trial() == second_row[0].item
        assert leafs[1].get_best_trial() == second_row[0].item
        assert leafs[2].get_best_trial() == second_row[0].item
        assert leafs[3].get_best_trial() == second_row[0].item
        # But not for fifth branch
        assert leafs[4].get_best_trial() == root.item

    def test_get_best_trial_equality(self):
        root = build_full_tree(4)

        leafs = root.get_nodes_at_depth(3)
        assert leafs[0].item.id == "id-8"
        assert leafs[0].get_best_trial() == root.item

        # Return parent in case of equality, if they are all as good, we want the earliest one.
        root.children[0].item.objective.value = root.item.objective.value
        assert leafs[0].get_best_trial() == root.item

        # Make sure the second one is returned is root is not as good.
        root.item.objective.value += 1
        assert leafs[0].get_best_trial() == root.children[0].item

    def test_get_best_trial_across_jumps(self):
        root_a = build_full_tree(4, starting_objective=1)
        root_b = build_full_tree(4, starting_objective=10)

        a_leafs = root_a.get_nodes_at_depth(3)
        b_leafs = root_b.get_nodes_at_depth(3)
        assert b_leafs[0].get_best_trial() == root_b.item
        a_leafs[0].set_jump(b_leafs[0].parent)

        # Should look past jump of parent
        assert b_leafs[0].get_best_trial() == root_a.item
        # Should look past jump directly
        assert b_leafs[0].parent.get_best_trial() == root_a.item
        # Should look towards root, there is no jump between root and this node
        assert b_leafs[0].parent.parent.get_best_trial() == root_b.item

    def test_get_best_trial_broken_leaf(self):
        root = build_full_tree(4, starting_objective=1)

        leafs = root.get_nodes_at_depth(3)
        leafs[0].item.objective = None
        assert leafs[0].get_best_trial() == root.item


class TestLineages:
    def test_what(self):
        assert False


class TestTruncate:
    def test_truncate_trial_not_in_trials(self, space, trials):
        trial = space.sample(1, seed=2)[0]

        with pytest.raises(
            ValueError,
            match=f"Trial {trial.id} not included in list of completed trials.",
        ):
            truncate(numpy.random.RandomState(1), trial, trials)

    def test_truncate_non_completed_trials(self, space, trials):
        trial = space.sample(1, seed=2)[0]
        trials.append(trial)

        assert trial in trials

        with pytest.raises(
            ValueError,
            match=f"Trial {trial.id} not included in list of completed trials.",
        ):
            truncate(numpy.random.RandomState(1), trial, trials)

    def test_truncate_empty_pool(self, space, trials):
        selected_trial = truncate(
            numpy.random.RandomState(1), trials[-1], trials, candidate_pool_ratio=0.0001
        )

        assert selected_trial is None

    @pytest.mark.parametrize("candidate_pool_ratio", [0.2, 0.4, 0.8])
    def test_truncate_valid_choice(
        self, candidate_pool_ratio, space, trials, monkeypatch
    ):
        num_completed_trials = len(trials)
        valid_choices = numpy.arange(
            int(candidate_pool_ratio * num_completed_trials)
        ).tolist()
        selected_trial = trials[valid_choices[-1]]

        def mocked_choice(choices, *args, **kwargs):
            assert choices.tolist() == valid_choices
            return valid_choices[-1]

        rng = RNGStub()
        rng.choice = mocked_choice

        completed_trial_index = numpy.random.choice(range(len(trials)))
        completed_trial = trials[completed_trial_index]

        # Add non completed trials and shuffle the list to test it is filtered and sorted properly
        trials += space.sample(20, seed=2)
        numpy.random.shuffle(trials)

        trial = truncate(
            rng,
            completed_trial,
            trials,
            truncation_threshold=1,
            candidate_pool_ratio=candidate_pool_ratio,
        )

        assert trial is selected_trial

    @pytest.mark.parametrize("truncation_threshold", [0.2, 0.4, 0.8])
    def test_truncate_no_need(self, truncation_threshold, space, trials, monkeypatch):
        # Test than trial within threshold is not replaced
        # TODO: test for multiple threshold
        threshold_index = truncation_threshold * len(trials)
        selected_index = numpy.random.choice(numpy.arange(threshold_index))

        # TODO there will be a bug if int(truncation_threshold * len()) == 0.
        # TODO test (in another test) for  int(candidate_pool_ratio * len()) == 0.

        num_completed_trials = len(trials)
        valid_choices = numpy.arange(
            int(candidate_pool_ratio * num_completed_trials)
        ).tolist()
        selected_trial = trials[valid_choices[-1]]

        def mocked_choice(choices, *args, **kwargs):
            assert choices.tolist() == valid_choices
            return valid_choices[-1]

        rng = RNGStub()
        rng.choice = mocked_choice

        completed_trial_index = numpy.random.choice(range(len(trials)))
        completed_trial = trials[completed_trial_index]

        # Add non completed trials and shuffle the list to test it is filtered and sorted properly
        trials += space.sample(20, seed=2)
        numpy.random.shuffle(trials)

        trial = truncate(
            rng,
            completed_trial,
            trials,
            truncation_threshold=1,
            candidate_pool_ratio=candidate_pool_ratio,
        )


class TestPerturb:
    def test_perturb_real_factor(self):
        assert False

    def test_perturb_real_volatility_below(self):
        assert False

    def test_perturb_real_volatility_above(self):
        assert False

    def test_perturb_int_factor(self):
        assert False

    def test_perturb_int_volatility_below(self):
        assert False

    def test_perturb_int_volatility_above(self):
        assert False

    def test_perturb_int_no_duplicate_below(self):
        assert False

    def test_perturb_int_no_duplicate_above(self):
        assert False

    def test_perturb_int_no_out_if_dim(self):
        assert False

    def test_perturb_int_cat(self):
        assert False

    def test_perturb(self):
        assert False

    def test_perturb_hierarchical_params(self):
        assert False

    def test_perturb_with_invalid_dim(self):
        assert False


class TestResample:
    # TODO: Should we return flat params or not??
    def test_resample_probability(self):
        assert False


class TestResampleOrPerturb:
    def test_perturb_if_not_resample(self):
        assert False

    def test_perturb_if_not_resample_hierarchical(self):
        assert False


class TestPBT(BaseAlgoTests):
    algo_name = "pbt"
    config = {"seed": 123456}


# TestRandomSearch.set_phases([("random", 0, "space.sample")])
