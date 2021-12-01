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


def truncate_with_backtracking(
    rng,
    fidelity,
    trial,
    lineages,
    min_forking_population=5,
    truncation_threshold=0.2,
    candidate_pool_ratio=0.2,
    backtracking_tolerance=0.2,
):
    """
    backtracking_tolerance: float, optional
        TODO: rewrite how backtracking_tolerance is used.

        If the objective drops by ``backtracking_tolerance``% from one fidelity to another,
        the lineage will be dropped and the candidate to select for forking will come from
        best trials so far (across all fidelity levels observed so far).
        Comes from [1]. Default: 0.2.

    [1] Zhang, Baohe, Raghu Rajan, Luis Pineda, Nathan Lambert, André Biedenkapp, Kurtland Chua,
    Frank Hutter, and Roberto Calandra. "On the importance of hyperparameter optimization for
    model-based reinforcement learning." In International Conference on Artificial Intelligence and
    Statistics, pp. 4015-4023. PMLR, 2021.
    """

    elites = lineages.get_elites()

    if len(elites) < min_forking_population:
        return None

    # TODO: If we compare to elites at any fidelity, then we will likely always
    # jump from trials at low fidelity if we have less workers than population_size.
    # We should compare to same fidelity, but jump to any fidelity.
    # This should documented because it differs from Zhang's paper.
    best_objective = min(elites.objective for elite in elites)
    if (
        get_objective(trial) - best_objective / numpy.abs(best_objective)
    ) > backtracking_tolerance:
        return random_choice(rng, elites, candidate_pool_ratio=candidate_pool_ratio)

    return truncate(
        rng,
        fidelity,
        trial,
        lineages,
        min_forking_population=min_forking_population,
        truncation_threshold=truncation_threshold,
        candidate_pool_ratio=candidate_pool_ratio,
    )


def truncate(
    rng,
    fidelity,
    trial,
    lineages,
    min_forking_population=5,
    truncation_threshold=0.2,
    candidate_pool_ratio=0.2,
):
    # TODO test if trial not in lineages?
    trial_nodes = lineages.get_nodes_at_depth(trial)
    completed_trials = [
        trial_node.item
        for trial_node in trial_nodes
        if trial_node.item.status == "completed"
    ]

    if len(completed_trials) < min_forking_population:
        return None

    sorted_trials = sorted(completed_trials, key=lambda trial: trial.objective.value)

    # Trial is good enough, PBT will re-use it.
    if trial not in sorted_trials[-int(truncation_threshold * len(sorted_trials)) :]:
        return trial

    return random_choice(rng, trials, candidate_pool_ratio=candidate_pool_ratio)


def random_choice(rng, trials, candidate_pool_ratio=0.2):
    sorted_trials = sorted(trials, key=lambda trial: trial.objective.value)

    if int(candidate_pool_ratio * len(sorted_trials)) == 0:
        return None

    index = rng.choice(numpy.arange(0, int(candidate_pool_ratio * len(sorted_trials))))
    return sorted_trials[index]


def perturb_real(rng, dim_value, interval, factor, volatility):
    if rng.random() > 0.5:
        dim_value *= factor
    else:
        dim_value *= 1.0 / factor

    if dim_value > interval[1]:
        dim_value = max(interval[1] - numpy.abs(rng.normal(0, volatility)), interval[0])
    elif dim_value < interval[0]:
        dim_value = min(interval[0] + numpy.abs(rng.normal(0, volatility)), interval[1])

    return dim_value


def perturb_int(rng, dim_value, interval, factor, volatility):
    new_dim_value = perturb_real(rng, dim_value, interval, factor, volatility)

    rounded_new_dim_value = int(numpy.round(new_dim_value))

    if rounded_new_dim_value == dim_value and new_dim_value > dim_value:
        new_dim_value = dim_value + 1
    elif rounded_new_dim_value == dim_value and new_dim_value < dim_value:
        new_dim_value = dim_value - 1
    else:
        new_dim_value = rounded_new_dim_value

    # Avoid out of dimension.
    new_dim_value = min(max(new_dim_value, interval[0]), interval[1])

    return new_dim_value


def perturb_cat(rng, dim_value, dim):
    return dim.sample(1, seed=tuple(rng.randint(0, 1000000, size=3)))[0]


def perturb(rng, trial, space, factor=1.2, volatility=0.0001):
    new_params = {}
    for dim in space.values():
        dim_value = flatten(trial.params)[dim.name]
        if dim.type == "real":
            dim_value = perturb_real(rng, dim_value, dim.interval(), factor, volatility)
        elif dim.type == "integer":
            dim_value = perturb_int(rng, dim_value, dim.interval(), factor, volatility)
        elif dim.type == "categorical":
            dim_value = perturb_cat(rng, dim_value, dim)
        elif dim.type == "fidelity":
            # do nothing
            pass
        else:
            raise ValueError(f"Unsupported dimension type {dim.type}")

        new_params[dim.name] = dim_value

    return new_params


def resample(rng, trial, space, probability=0.2):

    if probability > rng.uniform():
        trial = space.sample(1, seed=tuple(rng.randint(0, 1000000, size=3)))[0]

    return flatten(trial.params)


def resample_or_perturb(rng, trial, space, resample_kwargs, perturb_kwargs):
    params = resample(rng, trial, space, **resample_kwargs)

    if params != flatten(trial.params):
        return params

    return perturb(rng, trial, space, **perturb_kwargs)


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
    exploit: str or None, optional
        In the mutate part, one can define the customized mutate function with its mutate factors,
        such as multiply factor (times/divides by a multiply factor) and add factor
        (add/subtract by a multiply factor). The function must be defined by
        an importable string. If None, default
        mutate function is used: ``orion.algo.mutate_functions.default_mutate``.
    exploit_kwargs: dict or None, optional
        Arguments for the exploit function.
        TODO add info for default function.
    explore: str or None, optional
        In the mutate part, one can define the customized mutate function with its mutate factors,
        such as multiply factor (times/divides by a multiply factor) and add factor
        (add/subtract by a multiply factor). The function must be defined by
        an importable string. If None, default
        mutate function is used: ``orion.algo.mutate_functions.default_mutate``.
    explore_kwargs: dict or None, optional
        Arguments for the explore function.
        TODO add info for default function.


    """

    requires_type = None
    requires_dist = "linear"
    requires_shape = "flattened"

    def __init__(self, space, seed=None):
        super(PopulationBasedTraining, self).__init__(space, seed=seed)

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

        self.exploit_func = functools.partial(
            load_function(self.exploit), **self.exploit_kwargs
        )
        self.explore_func = functools.partial(
            load_function(self.explore), **self.explore_kwargs
        )

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
        for node in self.lineages.get_nodes_at_depth(final_depth):
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
                new_params = self.explore(self.rng, self.space, trial_to_explore.params)
                trial_to_branch = trial_to_explore

            # Set next level of fidelity
            new_params[self.fidelity_index] = self.fidelity_upgrades[
                trial_to_branch.params[self.fidelity_index]
            ]

            new_trial = trial_to_branch.branch(params=params)

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
                trial_to_retry = self.lineages.get_true_ancestor(trial)
                if trial_to_retry:
                    self._buffer.append(trial_to_retry)

            elif trial.status == "completed":
                self._buffer.append(trial)


class Lineages:
    def __init__(self):
        self._lineage_roots = []
        self._trial_to_lineages = {}

    def __iter__(self):
        return self._lineage_roots

    def add(self, trial):
        lineage = Lineage(trial)
        self._lineage_roots.append(lineage)
        self._trial_to_lineages[trial.id] = lineage
        return lineage

    def fork(self, base_trial, new_trial):
        new_lineage = self.get_lineage(base_trial).fork(new_trial)
        self._trial_to_lineages[new_trial.id] = new_lineage
        return new_lineage

    def set_jump(self, base_trial, new_trial):
        self.get_lineage(base_trial).set_jump(self.get_lineage(new_trial))

    def register(self, trial):
        if trial.id not in self._trial_to_lineages:
            lineage = self.add(trial)
        else:
            lineage = self.get_lineage(trial)
            lineage.register(trial)

        return lineage

    def get_lineage(self, trial):
        return self._trial_to_lineages[trial.id]

    def get_elites(self):
        trials = []
        for lineage in self._lineage_roots:
            for node in lineage.leafs:
                trials.append(node.get_best_trial())

        return trials

    def get_nodes_at_depth(self, trial_or_depth):
        if isinstance(trial_or_depth, int):
            depth = trial_or_depth
        else:
            depth = self.get_lineage(trial_or_depth).node_depth

        trial_nodes = []
        for lineage in self._lineage_roots:
            for trial_node in lineage.get_nodes_at_depth(depth):
                trial_nodes.append(trial_node)

        return trial_nodes

    def get_true_ancestor(self, trial):
        """
        note: return a trial, not a lineage
        """

        lineage = self.get_lineage(trial)
        if lineage.base is not None:
            return lineage.base.item

        if lineage.parent is not None:
            return lineage.parent.item

        return None


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

        shutil.copytree(self.item.working_dir, new_trial.working_dir)

        return Lineage(new_trial, parent=self)

    def set_jump(self, node):
        if node._jump.parent is not None:
            raise RuntimeError(
                "Trying to jump to an existing node. Jumps to another lineage should only "
                "occur on new nodes."
            )

        node._jump.set_parent(self._jump)

    def get_best_trial(self):
        # NOTE: best trial up to this node. Only looking towards parents (or jumps)
        parent_node = None
        if self.base is not None:
            parent_node = self.base
        elif self.parent is not None:
            parent_node = self.parent

        if parent_node:
            parent_trial = parent_node.get_best_trial()

            if get_objective(parent_trial) <= get_objective(self.item):
                return parent_trial

        return self.item
