# -*- coding: utf-8 -*-
"""
Population Based Training
=========================

"""
import copy
import shutil

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.random import Random
from orion.core.utils.tree import TreeNode
from orion.algo.pbt.exploit import exploit_factory
from orion.algo.pbt.explore import explore_factory
from orion.core.utils.flatten import flatten, unflatten


def get_objective(trial):
    if trial.objective and trial.objective.value is not None:
        return trial.objective.value

    return float("inf")


def compute_fidelities(n_branching, low, high, base):

    if base == 1:
        return numpy.linspace(low, high, num=n_branching + 1, endpoint=True)
    else:

        budgets = numpy.logspace(
            numpy.log(low) / numpy.log(base),
            numpy.log(high) / numpy.log(base),
            n_branching + 1,
            base=base,
            endpoint=True,
        )

        return budgets


class PopulationBasedTraining(BaseAlgorithm):
    """Population Based Training algorithm

    TODO
    Explain how to find working dir and how to set it.
    TODO
    Document how broken trials are handled

    Warn user that they should use trial.id for the working dir. Not hash-params. It will be copied
    by PBT anyway.

    Warn user that all trials should be using the same base working dir for the experiment.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    population_size: int, optional
        Size of the population. No trial will be continued until there are `population_size`
        trials executed until lowest fidelity. If a trial is broken during execution at lowest
        fidelity, the algorithm will sample a new trial, keeping the population of *non-broken*
        trials at `population_size`.  For efficiency it is better to have less workers running than
        total population_size. Default: 50.
    min_forking_population: int, optional
        Minimum number of trials completed at a given fidelity level to proceed with forking.
        If there are less than `min_forking_population` completed, the algorithm will wait.
        This ensures that forking are done when there is enough trial candidates to make a valuable
        forking.  Default: 5
    exploit: dict or None, optional
        In the mutate part, one can define the customized mutate function with its mutate factors,
        such as multiply factor (times/divides by a multiply factor) and add factor
        (add/subtract by a multiply factor). The function must be defined by
        an importable string. If None, default
        mutate function is used: ``orion.algo.mutate_functions.default_mutate``.
    explore: dict or None, optional
        In the mutate part, one can define the customized mutate function with its mutate factors,
        such as multiply factor (times/divides by a multiply factor) and add factor
        (add/subtract by a multiply factor). The function must be defined by
        an importable string. If None, default
        mutate function is used: ``orion.algo.mutate_functions.default_mutate``.

    """

    requires_type = None
    requires_dist = "linear"
    requires_shape = "flattened"

    def __init__(self, space, seed=None, exploit=None, explore=None):
        if exploit is None:
            exploit = {
                "of_type": "PipelineExploit",
                "exploit_configs": [
                    {
                        "of_type": "BacktrackExploit",
                        "min_forking_population": 5,
                        "truncation_quantile": 0.9,
                        "candidate_pool_ratio": 0.2,
                    },
                    {
                        "of_type": "TruncateExploit",
                        "min_forking_population": 5,
                        "truncation_quantile": 0.8,
                        "candidate_pool_ratio": 0.2,
                    },
                ],
            }

        if explore is None:
            explore = {
                "of_type": "PipelineExplore",
                "explore_configs": [
                    {"of_type": "ResampleExplore", "probability": 0.2},
                    {"of_type": "PerturbExplore", "factor": 1.2, "volatility": 0.0001},
                ],
            }

        super(PopulationBasedTraining, self).__init__(
            space, seed=seed, exploit=exploit, explore=explore
        )

        self.random_search = Random(space)
        self._buffer = []

        fidelity_index = self.fidelity_index
        if fidelity_index is None:
            raise RuntimeError(SPACE_ERROR)

        self.fidelity_dim = space.values()[fidelity_index]

        self.fidelities = compute_fidelities(
            self.n_branching, fidelity_dim.low, fidelity_dim.high, fidelity_dim.base
        )
        self.fidelity_upgrades = {a: b for a, b in zip(fidelities, fidelities[1:])}

        self.exploit_func = exploit_factory.create(**self.exploit)
        self.explore_func = explore_factory.create(**self.explore)

        self.lineages = []
        self._lineage_dropped_head = {}

    @property
    def space(self):
        """Return transformed space of PBT"""
        return self.random_search.space

    @space.setter
    def space(self, space):
        """Set the space of PBT and initialize it"""
        self.random_search.space = space

    @property
    def rng(self):
        return self.random_search.rng

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.random_search.seed_rng(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        _state_dict = super(PopulationBasedTraining, self).state_dict
        _state_dict["random_search"] = self.random_search.state_dict
        _state_dict["trials_children"] = self._trials_children
        return _state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict"""
        super(PopulationBasedTraining, self).set_state(state_dict)
        self.random_search.set_state(state_dict["random_search"])
        self._trials_children = state_dict["trials_children"]

    @property
    def num_root(self):
        return sum(int(lineage.root.status != "broken") for lineage in self.lineages)

    def is_done(self):
        # TODO: Take into account max cardinality.

        n_completed = 0
        final_depth = self.get_depth_of(self.fidelity_dim.high)
        for trial in self.lineages.get_trials_at_depth(final_depth):
            n_completed += int(node.status == "completed")

        return n_completed >= self.population_size

    def register(self, trial):
        super(PopulationBasedTraining, self).register(trial)
        self.lineages.register(trial)

    def suggest(self, num):

        # Sample points until num is met, or population_size
        trials = self.sample(num)

        # Then try branching based on observed_buffer until num is met or buffer is exhausted.
        trials += self.fork_lineages(max(len(trials) - num, 0))

        return trials

    def sample(self, num):
        sampled_trials = self.random_search.suggest(
            min(max(self.population_size - self.num_root, 0), num)
        )

        trials = []
        for trial in sampled_trials:
            branched_trial = trial.branch(
                params={self.fidelity_dim.name: self.fidelity_dim.low}
            )
            self.register(branched_trial)
            trials.append(branched_trial)

        return trials

    def get_depth_of(self, fidelity):
        return self.fidelities.index(fidelity)

    def fork_lineages(self, num):

        branched_trials = []
        skipped = []

        while len(branched_trials) < num and self._buffer:
            trial = self._buffer.pop(0)

            trial_to_branch, new_trial = self.generate_offspring(trial)

            if trial_to_branch is None:
                skipped_trials.append(trial)
                continue

            self.lineages.fork(trial_to_branch, new_trial)

            if base_trial is not trial_to_branch:
                self.lineages.set_jump(base_trial, new_trial)

            branched_trials.append(new_trial)

        self._buffer = skipped_trials + self._buffer

        return branched_trials

    def generate_offspring(self, trial, population):
        new_trial = trial

        if not self.has_suggested(new_trial):
            raise RuntimeError(
                "Trying to fork a trial that was not registered yet. This should never happen"
            )

        start = time.time()
        while (
            self.has_suggested(new_trial) and time.time() - start <= self.fork_timeout
        ):
            trial_to_explore = self.exploit_func(
                self.rng,
                trial,
                self.lineages,
            )

            if trial_to_explore is None:
                return None, None
            elif trial_to_explore is trial:
                new_params = {}
                trial_to_branch = trial
            else:
                new_params = flatten(
                    self.explore_func(self.rng, self.space, trial_to_explore.params)
                )
                trial_to_branch = trial_to_explore

            # Set next level of fidelity
            new_params[self.fidelity_index] = self.fidelity_upgrades[
                trial_to_branch.params[self.fidelity_index]
            ]

            new_trial = trial_to_branch.branch(params=new_params)

        if self.has_suggested(new_trial) and time.time() - start > self.fork_timeout:
            raise SuggestionTimeout()

        return trial_to_branch, new_trial

    def adopt(self, trial):
        parent = self._trials_info.get(trial.parent, None)
        if flatten(trial.params)[self.fidelity_index] == self.fidelities[0]:
            # Add to lineages as root.
            adopted = True
        elif parent and self.has_observed(parent):
            # Add child to corresponding lineage, no fork with copy of folder
            adopted = True
        else:
            log.info(f"Unknown trial lineage, cannot adopt: {trial.id}")
            adopted = False

        return adopted

    def observe(self, trials):
        # TODO: Need to handle resumption. How do we rebuild the tree?

        trials_to_verify = []

        # First try to resume from trials if necessary, then only push to buffer leafs
        for trial in trials:
            if not self.has_suggested(trial):
                adopted = self.adopt(trial)
                if adopted:
                    trials_to_verify.append(trial)
            elif not self.has_observed(trial):
                self.register(trial)
                trials_to_verify.append(trial)

        for trial in trials_to_verify:
            if self.lineages.get_lineage(trial).children:
                continue

            # TODO: On resumption, broken trials will be observed and will lead
            # to retry
            if trial.status == "broken":
                # Branch again from trial that lead to this broken one.
                trial_to_retry = self.lineages.get_lineage(trial).get_true_ancestor()
                if trial_to_retry:
                    self._buffer.append(trial_to_retry)

            elif trial.status == "completed":
                self._buffer.append(trial)


class Lineages:
    def __init__(self):
        self._lineage_roots = []
        self._trial_to_lineages = {}

    def __len__(self):
        return len(self._lineage_roots)

    def __iter__(self):
        return iter(self._lineage_roots)

    def add(self, trial):
        if trial.id in self._trial_to_lineages:
            return self._trial_to_lineages[trial.id]

        lineage = Lineage(trial)
        self._lineage_roots.append(lineage)
        self._trial_to_lineages[trial.id] = lineage
        return lineage

    def fork(self, base_trial, new_trial):
        new_lineage = self._trial_to_lineages[base_trial.id].fork(new_trial)
        self._trial_to_lineages[new_trial.id] = new_lineage
        return new_lineage

    def get_lineage(self, trial):
        """
        Raises
        ------
        KeyError
        """
        return self._trial_to_lineages[trial.id]

    def set_jump(self, base_trial, new_trial):
        self.get_lineage(base_trial).set_jump(self.get_lineage(new_trial))

    def register(self, trial):
        if trial.id not in self._trial_to_lineages:
            lineage = self.add(trial)
        else:
            lineage = self.get_lineage(trial)
            lineage.register(trial)

        return lineage

    def get_elites(self, max_depth=None):
        trials = []
        for lineage in self._lineage_roots:
            # TODO: That does not work. We need to go bottom up, and keep on one line.
            #       Problem is, there may be multiple jumps. How to know which one to follow?
            #       No, there may be several forks, but only 1 jump...
            if max_depth is None:
                nodes = lineage.leafs
            else:
                nodes = lineage.get_nodes_at_depth(max_depth)

            for node in nodes:
                if node.jumps and (
                    (max_depth is None) or (node.node_depth < max_depth)
                ):
                    continue

                best_trial = node.get_best_trial()
                if best_trial is not None:
                    trials.append(best_trial)

        return trials

    def get_trials_at_depth(self, trial_or_depth):
        if isinstance(trial_or_depth, int):
            depth = trial_or_depth
        else:
            depth = self.get_lineage(trial_or_depth).node_depth

        trials = []
        for lineage in self._lineage_roots:
            for trial_node in lineage.get_nodes_at_depth(depth):
                trials.append(trial_node.item)

        return trials


class Lineage(TreeNode):
    """
    TODO: Document the additional feature jump/base
    """

    def __init__(self, trial, parent=None):
        super(Lineage, self).__init__(copy.deepcopy(trial), parent=parent)
        self._jump = TreeNode(self)

    @property
    def tree_name(self):
        return str(self.item)

    @property
    def jumps(self):
        return [node.item for node in self._jump.children]

    @property
    def base(self):
        return self._jump.parent.item if self._jump.parent else None

    def register(self, trial):
        self.item = copy.deepcopy(trial)

    def fork(self, new_trial):
        if self.item.working_dir == new_trial.working_dir:
            raise RuntimeError(
                f"The new trial {new_trial.id} has the same working directory as "
                f"trial {self.item.id}, which would lead to corrupted checkpoints. "
                "This should never happen. Please "
                "report at https://github.com/Epistimio/orion/issues"
            )

        try:
            shutil.copytree(self.item.working_dir, new_trial.working_dir)
        except FileExistsError as e:
            raise FileExistsError(
                f"Folder already exists for trial {new_trial.id}. This could be a folder "
                "remaining from a previous experiment with same trial id."
            ) from e

        return Lineage(new_trial, parent=self)

    def set_jump(self, node):
        if node._jump.parent is not None:
            raise RuntimeError(
                "Trying to jump to an existing node. Jumps to another lineage should only "
                "occur on new nodes."
            )

        node._jump.set_parent(self._jump)

    def get_true_ancestor(self):
        if self.base is not None:
            return self.base

        if self.parent is not None:
            return self.parent

        return None

    def get_best_trial(self):
        # NOTE: best trial up to this node. Only looking towards parents (or jumps)
        parent_node = self.get_true_ancestor()

        if parent_node:
            parent_trial = parent_node.get_best_trial()

            if get_objective(parent_trial) <= get_objective(self.item):
                return parent_trial

        if self.item.status != "completed":
            return None

        return self.item
