import os
import random

import pytest
from base import (
    ObjectiveStub,
    TrialStub,
    build_full_tree,
    build_population,
    compare_generations,
)

from orion.algo.pbt.pbt import LineageNode, Lineages


class TestLineageNode:
    def test_register(self):
        item = [0]
        lineage = LineageNode(item)
        assert lineage.item == item
        assert lineage.item is not item

        item = [1]
        lineage.register(item)
        assert lineage.item == item
        assert lineage.item is not item

    def test_fork(self):
        path = "/some_path"
        trial = TrialStub(path)
        lineage = LineageNode(trial)

        new_path = "/another_path"
        new_trial = TrialStub(new_path)

        new_lineage = lineage.fork(new_trial)

        assert new_lineage.item.working_dir == new_trial.working_dir
        assert new_lineage.parent is lineage
        assert lineage.children[0] is new_lineage

    def test_set_jump(self):
        parent_lineage = LineageNode(1)
        child_lineage = LineageNode(2)
        parent_lineage.set_jump(child_lineage)

        assert child_lineage.parent is None
        assert child_lineage.jumps == []
        assert child_lineage.base is parent_lineage

        assert parent_lineage.children == []
        assert parent_lineage.jumps == [child_lineage]
        assert parent_lineage.base is None

    def test_set_jump_twice(self):
        parent_lineage = LineageNode(1)
        child_lineage = LineageNode(2)
        parent_lineage.set_jump(child_lineage)

        another_child_lineage = LineageNode(3)
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
        parent_lineage = LineageNode(1)
        child_lineage = LineageNode(2)
        parent_lineage.set_jump(child_lineage)

        another_child_lineage = LineageNode(3)

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

    def test_get_true_ancestor_no_parent(self):
        lineage = LineageNode(1)
        assert lineage.get_true_ancestor() is None

    def test_get_true_ancestor_parent_no_jump(self):
        lineage = LineageNode(1)
        child_lineage = LineageNode(2, parent=lineage)
        assert child_lineage.get_true_ancestor() is lineage

    def test_get_true_ancestor_with_jump(self):
        lineage = LineageNode(1)
        child_lineage = LineageNode(2, parent=lineage)
        true_lineage = LineageNode(3)
        true_lineage.set_jump(child_lineage)
        assert child_lineage.parent is lineage
        assert child_lineage.base is true_lineage
        assert child_lineage.get_true_ancestor() is true_lineage

    def test_get_best_trial_empty(self):
        trial = TrialStub(id="id-1", objective=1)
        lineage = LineageNode(trial)
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

    def test_get_best_trial_non_completed_root(self):
        lineage = LineageNode(TrialStub(id="my-id"))
        assert lineage.get_best_trial() is None


class TestLineages:
    def test_add_new_trial(self):
        lineages = Lineages()
        assert len(lineages) == 0
        lineage = lineages.add(TrialStub(id="stub"))
        assert len(lineages) == 1
        assert lineages._lineage_roots[0] is lineage
        assert lineages._trial_to_lineages["stub"] is lineage

    def test_add_duplicate(self):
        lineages = Lineages()
        assert len(lineages) == 0
        lineage = lineages.add(TrialStub(id="stub"))
        assert len(lineages) == 1

        new_lineage = lineages.add(TrialStub(id="stub"))
        assert new_lineage is lineage
        assert len(lineages) == 1

    def test_fork_existing_trial(self, tmp_path):
        lineages = Lineages()
        trial = TrialStub(id="stub", working_dir=os.path.join(tmp_path, "stub"))
        os.makedirs(trial.working_dir)
        lineage = lineages.add(trial)
        assert len(lineages) == 1
        new_trial = TrialStub(id="fork", working_dir=os.path.join(tmp_path, "fork"))
        new_lineage = lineages.fork(trial, new_trial)
        assert len(lineages) == 1
        assert lineages._lineage_roots[0].children[0] is new_lineage
        assert lineages._trial_to_lineages["fork"] is new_lineage

    def test_fork_non_existing_trial(self):
        lineages = Lineages()
        trial = TrialStub(id="stub")
        new_trial = TrialStub(id="fork")

        with pytest.raises(KeyError):
            new_lineage = lineages.fork(trial, new_trial)

    def test_get_lineage_existing_root_trial(self):
        lineages = Lineages()
        trial = TrialStub(id="stub")
        lineage = lineages.add(trial)
        assert lineages.get_lineage(trial) is lineage

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_get_lineage_existing_node_trial(self):
        lineages = Lineages()
        for root_index in range(2):

            trial = TrialStub(id=f"lineage-{root_index}-0")
            lineage = lineages.add(trial)
            for depth in range(1, 10):
                new_trial = TrialStub(id=f"lineage-{root_index}-{depth}")
                lineage = lineages.fork(trial, new_trial)
                trial = new_trial

        lineage = lineages.get_lineage(TrialStub(id="lineage-0-2"))
        assert lineage.root is lineages._lineage_roots[0]
        assert lineage.node_depth == 2

        lineage = lineages.get_lineage(TrialStub(id="lineage-1-5"))
        assert lineage.root is lineages._lineage_roots[1]
        assert lineage.node_depth == 5

    def test_get_lineage_non_existing_trial(self):
        lineages = Lineages()

        with pytest.raises(KeyError):
            lineages.get_lineage(TrialStub(id="id"))

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_set_jump_existing_trial(self):
        lineages = Lineages()
        root_1 = TrialStub(id="root-1")
        lineage_1 = lineages.add(root_1)
        root_2 = TrialStub(id="root-2")
        lineage_2 = lineages.add(root_2)
        child_trial = TrialStub(id="child")
        child_lineage = lineages.fork(root_1, child_trial)
        lineages.set_jump(root_2, child_trial)

        assert child_lineage.base is lineage_2
        assert lineage_2.jumps == [child_lineage]
        assert child_lineage.jumps == []
        assert lineage_2.base is None
        assert lineage_1.jumps == []
        assert lineage_1.base is None

    def test_set_jump_non_existing_base_trial(self):
        lineages = Lineages()
        with pytest.raises(KeyError, match="'dontexist'"):
            lineages.set_jump(
                TrialStub(id="dontexist"), TrialStub(id="dontexistbutdoesntmatter")
            )

    def test_set_jump_non_existing_new_trial(self):
        lineages = Lineages()
        trial = TrialStub(id="exists")
        lineages.add(trial)
        with pytest.raises(KeyError, match="'newtrialdontexist'"):
            lineages.set_jump(trial, TrialStub(id="newtrialdontexist"))

    def test_register_new_trial(self):
        lineages = Lineages()
        new_trial = TrialStub(id="new")
        lineage = lineages.register(new_trial)
        assert lineages._lineage_roots == [lineage]

    def test_register_existing_trial(self):
        lineages = Lineages()
        trial = TrialStub(id="my-id")
        lineage = lineages.add(trial)
        assert lineages._lineage_roots == [lineage]
        assert lineage.item.objective is None

        trial.objective = ObjectiveStub(1)
        assert lineages.register(trial) is lineage
        assert lineages._lineage_roots == [lineage]
        assert lineage.item.objective.value == 1

    def test_get_elites_empty(self):
        lineages = Lineages()
        assert lineages.get_elites() == []

    def test_get_elites_none_completed(self):
        lineages = Lineages()
        lineages.add(TrialStub(id="1"))
        lineages.add(TrialStub(id="2"))
        lineages.add(TrialStub(id="3"))
        assert lineages.get_elites() == []

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_get_elites_various_depths(self):

        lineages = build_population(
            [
                [2, 8, 9, 9, 8],
                [9, 3, 8, 9, 7],
                [8, 8, 8, 4, 6],
                [7, 8, 9, 8, 5],
                [7, 6, 8, 7, 5],
                [6, 5, 7, 7, 4],
                [5, 5, 6, 7, 5],
                [4, 4, 5, 8, 5],
                [4, 4, 9, 8, 5],
                [4, 4, 8, 8, 5],
                [4, 4, 7, 8, 5],
                [4, 4, 6, 8, 5],
                [4, 4, 8, 8, 5],
                [4, 4, 9, 8, 5],
            ]
        )

        elites = sorted(lineages.get_elites(), key=lambda trial: trial.id)
        assert len(elites) == 5
        assert elites[0].id == "lineage-0-0"
        assert elites[0].objective.value == 2

        assert elites[1].id == "lineage-1-1"
        assert elites[1].objective.value == 3

        assert elites[2].id == "lineage-2-7"
        assert elites[2].objective.value == 5

        assert elites[3].id == "lineage-3-2"
        assert elites[3].objective.value == 4

        assert elites[4].id == "lineage-4-5"
        assert elites[4].objective.value == 4

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_get_elites_max_depth(self):
        lineages = build_population(
            [
                [2, 8, 9, 9, 8],
                [9, 3, 8, 9, 7],
                [8, 8, 8, 4, 6],
                [7, 8, 9, 8, 5],
                [7, 6, 8, 7, 5],
                [6, 5, 7, 7, 4],
                [5, 5, 6, 7, 5],
                [4, 4, 5, 8, 5],
                [4, 4, 9, 8, 5],
                [4, 4, 8, 8, 5],
                [4, 4, 7, 8, 5],
                [4, 4, 6, 8, 5],
                [4, 4, 8, 8, 5],
                [4, 4, 9, 8, 5],
            ]
        )

        elites = sorted(lineages.get_elites(0), key=lambda trial: trial.id)
        assert [trial.objective.value for trial in elites] == [2, 8, 9, 9, 8]

        elites = sorted(lineages.get_elites(2), key=lambda trial: trial.id)
        assert [trial.objective.value for trial in elites] == [2, 3, 8, 4, 6]

        elites = sorted(lineages.get_elites(5), key=lambda trial: trial.id)
        assert [trial.objective.value for trial in elites] == [2, 3, 7, 4, 4]

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_get_trials_at_depth_given_depth(self):
        population_size = 5
        generations = 10
        lineages = build_population(
            [list(range(population_size)) for generation in range(generations)]
        )
        for depth in [0, 1, 5, 9]:
            compare_generations(
                lineages.get_trials_at_depth(depth), population_size, depth
            )

        assert lineages.get_trials_at_depth(10) == []

    @pytest.mark.usefixtures("no_shutil_copytree")
    def test_get_trials_at_depth_given_existing_trial(self):
        population_size = 5
        generations = 10
        lineages = build_population(
            [list(range(population_size)) for generation in range(generations)]
        )
        for depth in [0, 1, 5, 9]:
            lineage_index = random.choice(range(population_size))
            trial = TrialStub(id=f"lineage-{lineage_index}-{depth}")
            compare_generations(
                lineages.get_trials_at_depth(trial), population_size, depth
            )

    def test_get_trials_at_depth_given_non_existing_trial(self):
        lineages = Lineages()

        with pytest.raises(KeyError, match="idontexist"):
            lineages.get_trials_at_depth(TrialStub(id="idontexist"))
