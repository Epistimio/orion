"""
Population Based Training
=========================

"""
from __future__ import annotations

import copy
import logging
import time
from typing import Any, ClassVar, Iterable, Sequence

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.pbt.exploit import exploit_factory
from orion.algo.pbt.explore import explore_factory
from orion.algo.random import Random
from orion.algo.space import Fidelity, Space
from orion.core.utils.flatten import flatten
from orion.core.utils.tree import TreeNode
from orion.core.worker.transformer import TransformedDimension
from orion.core.worker.trial import Trial

logger = logging.getLogger(__name__)


SPACE_ERROR = """
PBT cannot be used if space does not contain a fidelity dimension.
For more information on the configuration and usage of Hyperband, see
https://orion.readthedocs.io/en/develop/user/algorithms.html#pbt
"""


def get_objective(trial: Trial) -> float:
    if trial.objective and trial.objective.value is not None:
        return trial.objective.value

    return float("inf")


def compute_fidelities(
    n_branching: int,
    low: numpy.ndarray | float,
    high: numpy.ndarray | float,
    base: float,
) -> list[float]:
    if base == 1:
        fidelities = numpy.linspace(low, high, num=n_branching + 1, endpoint=True)
    else:
        fidelities = numpy.logspace(
            numpy.log(low) / numpy.log(base),
            numpy.log(high) / numpy.log(base),
            n_branching + 1,
            base=base,
            endpoint=True,
        )

    return numpy.clip(fidelities, a_min=low, a_max=high).tolist()


class PBT(BaseAlgorithm):
    """Population Based Training algorithm

    Warning:PBT was broken in version v0.2.4. Make sure to use latest release.

    Population based training is an evolutionary algorithm that evolve trials
    from low fidelity levels to high fidelity levels (ex: number of epochs).
    For a population of size `m`, it first samples `m` trials at lowest fidelity level.
    When trials are completed, it decides based on the ``exploit`` configuration whether
    the trial should be promoted to next fidelity level or whether another trial
    should be selected instead and forked. When a trial is forked, new hyperparameters are
    selected based on the trials hyperparameters and the ``explore`` configuration.
    The original trial's working_dir is then copied over to the new trial's working_dir
    so that the user script can resume execution from model parameters of original trial.

    It is important that the weights of models trained for each trial are saved in the corresponding
    directory at path ``trial.working_dir``. The file name does not matter. The entire directory is
    copied to a new ``trial.working_dir`` when PBT selects a good model and explore new
    hyperparameters. The new trial can be resumed by the user by loading the weights found in the
    freshly copied ``new_trial.working_dir``, and saved back at the same path at end of trial
    execution. To access ``trial.working_dir`` from Oríon's commandline API, see documentation at
    https://orion.readthedocs.io/en/stable/user/script.html#command-line-templating. To access
    ``trial.working_dir`` from Oríon's Python API, set argument ``trial_arg="trial"`` when executing
    method :py:meth:`orion.client.experiment.ExperimentClient.workon`.

    The number of fidelity levels is determined by the argument ``generations``. The lowest
    and highest fidelity levels, and the distrubition, is determined by the search space's
    dimension that will have a prior ``fidelity(low, high, base)``, where ``base`` is the
    logarithm base of the dimension. Original PBT algorithm uses a base of 1.

    PBT will try to return as many trials as possible when calling ``suggest(num)``, up to ``num``.
    When ``population_size`` trials are sampled and more trials are requested, it will try to
    generate new trials by promoting or forking existing trials in a queue. This queue will get
    filled when calling ``observe(trials)`` on completed or broken trials.

    If trials are broken at lowest fidelity level, they are ignored and will not count
    in population size so that PBT can sample additional trials to reach ``population_size``
    completed trials at lowest fidelity. If a trial is broken at higher fidelity, the
    original trial leading to the broken trial is examined again for ``exploit`` and ``explore``.
    If the broken trial was the result of a fork, then we backtrack to the trial that was dropped
    during ``exploit`` in favor of the forked trial. If the broken trial was a promotion, then
    we backtrack to the original trial that was promoted.

    For more information on the algorithm,
    see original paper at https://arxiv.org/abs/1711.09846.

    Jaderberg, Max, et al. "Population based training of neural networks."
    arXiv preprint, arXiv:1711.09846 (2017).

    Notes
    -----
    It is important that the experiment using this algorithm has a working directory properly
    set. The experiment's working dir serve as the base for the trial's working directories.

    The trial's working directory is ``trial.working_dir``. This is where the weights of the model
    should be saved. Using ``trial.hash_params`` to determine a unique working dir for the trial
    will result in working on a different directory than the one copied by PBT, hence missing the
    copied model parameters.

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
        population_size. Default: 50.
    generations: int, optional
        Number of generations, from lowest fidelity to highest one. This will determine how
        many branchings occur during the execution of PBT. Default: 10
    exploit: dict or None, optional
        Configuration for a ``pbt.exploit.BaseExploit`` object that determines
        when if a trial should be exploited or not. If None, default configuration
        is a ``PipelineExploit`` with ``BacktrackExploit`` and ``TruncateExploit``.
    explore: dict or None, optional
        Configuration for a ``pbt.explore.BaseExplore`` object that returns new parameter
        values for exploited trials. If None, default configuration is a ``PipelineExplore`` with
        ``ResampleExplore`` and ``PerturbExplore``.
    fork_timeout: int, optional
        Maximum amount of time in seconds that an attempt to mutate a trial should take, otherwise
        algorithm.suggest() will raise ``SuggestionTimeout``. Default: 60

    """

    requires_type: ClassVar[str | None] = None
    requires_dist: ClassVar[str] = "linear"
    requires_shape: ClassVar[str] = "flattened"

    def __init__(
        self,
        space: Space,
        seed: int | Sequence[int] | None = None,
        population_size: int = 50,
        generations: int = 10,
        exploit: dict | None = None,
        explore: dict | None = None,
        fork_timeout: int = 60,
    ):
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

        self.random_search = Random(space)
        self._queue: list[Trial] = []

        fidelity_index = self.fidelity_index
        if fidelity_index is None:
            raise RuntimeError(SPACE_ERROR)

        fidelity_dim = space[fidelity_index]
        while isinstance(fidelity_dim, TransformedDimension):
            fidelity_dim = fidelity_dim.original_dimension
        assert isinstance(fidelity_dim, Fidelity)
        self.fidelity_dim = fidelity_dim

        self.fidelities = compute_fidelities(
            generations,
            self.fidelity_dim.low,
            self.fidelity_dim.high,
            self.fidelity_dim.base,
        )
        self.fidelity_upgrades = dict(zip(self.fidelities, self.fidelities[1:]))
        logger.info("Executing PBT with fidelities: %s", self.fidelities)

        self.exploit_func = exploit_factory.create(**exploit)
        self.explore_func = explore_factory.create(**explore)

        self.lineages = Lineages()

        super().__init__(space)
        self.seed = seed
        self.population_size = population_size
        self.generations = generations
        self.exploit = exploit
        self.explore = explore
        self.fork_timeout = fork_timeout
        if seed is not None:
            self.seed_rng(seed=seed)

    @property
    def space(self) -> Space:
        """Return transformed space of PBT"""
        return self.random_search.space

    @space.setter
    def space(self, space: Space) -> None:
        """Set the space of PBT and initialize it"""
        self.random_search.space = space

    @property
    def rng(self) -> numpy.random.RandomState:
        """Random Number Generator"""
        return self.random_search.rng

    def seed_rng(self, seed: int | Sequence[int] | None) -> None:
        """Seed the state of the random number generator.

        Parameters
        ----------
        seed: int
            Integer seed for the random number generator.
        """
        self.random_search.seed_rng(seed)

    @property
    def state_dict(self) -> dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict: dict[str, Any] = super().state_dict
        state_dict["random_search"] = self.random_search.state_dict
        state_dict["lineages"] = copy.deepcopy(self.lineages)
        state_dict["queue"] = copy.deepcopy(self._queue)
        return state_dict

    def set_state(self, state_dict: dict) -> None:
        """Reset the state of the algorithm based on the given state_dict"""
        super().set_state(state_dict)
        self.random_search.set_state(state_dict["random_search"])
        self.lineages = state_dict["lineages"]
        self._queue = state_dict["queue"]

    @property
    def _num_root(self) -> int:
        """Number of trials with lowest fidelity level that are not broken."""
        return sum(
            int(lineage.root.item.status != "broken") for lineage in self.lineages
        )

    @property
    def is_done(self) -> bool:
        """Is done if ``population_size`` trials at highest fidelity level are completed."""
        n_completed = 0
        final_depth = self._get_depth_of(self.fidelities[-1])
        for trial in self.lineages.get_trials_at_depth(final_depth):
            n_completed += int(trial.status == "completed")

        return n_completed >= self.population_size

    def register(self, trial: Trial) -> None:
        """Save the trial as one suggested or observed by the algorithm

        The trial is additionally saved in the lineages object of PBT.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
            Trial from a `orion.algo.space.Space`.

        """
        super().register(trial)
        self.lineages.register(trial)

    def suggest(self, num: int) -> list[Trial]:
        """Suggest a ``num`` ber of new sets of parameters.

        PBT will try to sample up to ``population_size`` trials at lowest fidelity level.
        If more trials are required, it will try to promote or fork trials based on the queue
        of available trials observed.

        Parameters
        ----------
        num: int
            Number of points to suggest. The algorithm may return less than the number of points
            requested.

        Returns
        -------
        list of trials
            A list of trials representing values suggested by the algorithm.

        """

        # Sample points until num is met, or population_size
        num_random_samples = min(max(self.population_size - self._num_root, 0), num)
        logger.debug(
            "PBT has %s pending or completed trials at root, %s broken trials.",
            self._num_root,
            len(self.lineages) - self._num_root,
        )
        logger.debug("Sampling %s new trials", num_random_samples)
        trials = self._sample(num_random_samples)
        logger.debug("Sampled %s new trials", len(trials))
        logger.debug(
            "After sampling, PBT has %s pending or completed trials at root, %s broken trials.",
            self._num_root,
            len(self.lineages) - self._num_root,
        )

        # Then try branching based on observed_queue until num is met or queue is exhausted.
        num_fork_samples = max(num - len(trials), 0)
        logger.debug(
            "Attempting Forking %s trials, with %s trials queued available for forking",
            num_fork_samples,
            len(self._queue),
        )
        forked_trials = self._fork_lineages(num_fork_samples)
        logger.debug("Forked %s new trials", len(forked_trials))
        logger.debug(
            "After forking, PBT has %s pending or completed trials at root, %s broken trials.",
            self._num_root,
            len(self.lineages) - self._num_root,
        )

        trials += forked_trials

        return trials

    def _sample(self, num: int) -> list[Trial]:
        """Sample trials based on random search"""
        sampled_trials = self.random_search.suggest(num)

        trials = []
        for trial in sampled_trials:
            branched_trial = trial.branch(
                params={self.fidelity_dim.name: self.fidelity_dim.low}
            )
            # NOTE: We are using branching as a simple way to update the fidelity, we should not keep the
            #       parent id since it is not a mutation from a parent trial.
            branched_trial.parent = None
            self.register(branched_trial)
            trials.append(branched_trial)

        return trials

    def _get_depth_of(self, fidelity: Any) -> int:
        """Get the depth of a fidelity in the lineages"""
        # NOTE: There is an issue with rounding, asking for fidelity of `10` but list has
        # `10.00000004`. This is a hacky bugfix.
        if fidelity in self.fidelities:
            return self.fidelities.index(fidelity)
        elif fidelity in numpy.round(self.fidelities, decimals=4):
            return numpy.round(self.fidelities, decimals=4).tolist().index(fidelity)
        raise RuntimeError(
            f"Fidelity {fidelity} not found in the fidelities {self.fidelities}."
        )

    def _fork_lineages(self, num: int) -> list[Trial]:
        """Try to promote or fork up to ``num`` trials from the queue."""

        branched_trials: list[Trial] = []
        skipped_trials: list[Trial] = []

        while len(branched_trials) < num and self._queue:
            trial = self._queue.pop(0)

            trial_to_branch, new_trial = self._generate_offspring(trial)
            if trial_to_branch is None:
                logger.debug("Skipping trial %s", trial)
                skipped_trials.append(trial)
                continue
            # NOTE: new_trial isn't None if trial_to_branch is not None.
            assert new_trial is not None
            self.lineages.fork(trial_to_branch, new_trial)

            if trial is not trial_to_branch:
                logger.debug("Dropped trial %s in favor of %s", trial, trial_to_branch)
                self.lineages.set_jump(trial, new_trial)

            logger.debug("Forking trial %s to %s", trial_to_branch, new_trial)

            branched_trials.append(new_trial)
            self.register(new_trial)

        self._queue = skipped_trials + self._queue

        return branched_trials

    def _generate_offspring(
        self, trial: Trial
    ) -> tuple[Trial, Trial] | tuple[None, None]:
        """Try to promote or fork a given trial."""

        new_trial = trial

        if not self.has_suggested(new_trial):
            raise RuntimeError(
                "Trying to fork a trial that was not registered yet. This should never happen"
            )

        attempts = 0
        start = time.perf_counter()

        trial_to_branch: Trial | None = None
        while (
            self.has_suggested(new_trial)
            and time.perf_counter() - start <= self.fork_timeout
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
                logger.debug("Promoting trial %s, parameters stay the same.", trial)
            else:
                new_params = flatten(
                    self.explore_func(self.rng, self.space, trial_to_explore.params)
                )
                trial_to_branch = trial_to_explore
                logger.debug(
                    "Forking trial %s with new parameters %s",
                    trial_to_branch,
                    new_params,
                )
            assert trial_to_branch is not None  # note: implicitly assumed below.
            # Set next level of fidelity
            new_params[self.fidelity_index] = self.fidelity_upgrades[
                trial_to_branch.params[self.fidelity_index]
            ]

            new_trial = trial_to_branch.branch(params=new_params)
            # TODO: Keep this? or not?
            assert new_trial.parent is not None

            logger.debug("Attempt %s - Creating new trial %s", attempts, new_trial)

            attempts += 1

        if (
            self.has_suggested(new_trial)
            and time.perf_counter() - start > self.fork_timeout
        ):
            trial_to_branch = None
            new_trial = None
            logger.info(
                f"Could not generate unique new parameters for trial {trial.id} in "
                f"less than {self.fork_timeout} seconds. Attempted {attempts} times."
            )

        return trial_to_branch, new_trial

    def _triage(self, trials: list[Trial]) -> list[Trial]:
        """Triage observed trials and return those that may be queued."""

        trials_to_verify: list[Trial] = []

        # First try to resume from trials if necessary, then only push to queue leafs
        for trial in trials:
            if not self.has_suggested(trial):
                logger.debug("Ignoring unknown trial %s", trial)
                continue

            if not self.has_observed(trial) and trial.status in ["completed", "broken"]:
                logger.debug("Will verify trial %s for queue", trial)
                trials_to_verify.append(trial)

            self.register(trial)

        return trials_to_verify

    def _queue_trials_for_promotions(self, trials: list[Trial]) -> None:
        """Queue trials if they are completed or ancestor trials if they are broken."""
        for trial in trials:
            if trial.status == "broken":
                # Branch again from trial that lead to this broken one.
                lineage_to_retry = self.lineages.get_lineage(trial).get_true_ancestor()
                if lineage_to_retry:
                    logger.debug(
                        "Trial %s is broken, queuing ancestor %s to re-attempt forking.",
                        trial,
                        lineage_to_retry.item,
                    )
                    self._queue.append(lineage_to_retry.item)
                else:
                    logger.debug(
                        (
                            "Trial %s from initial generation is broken, "
                            "new trials can be sampled at next suggest() call."
                        ),
                        trial,
                    )

            elif trial.status == "completed":
                if trial.params[self.fidelity_index] == self.fidelities[-1]:
                    logger.debug(
                        "Trial %s is completed at full fidelity (%s). Not forking.",
                        trial,
                        self.fidelities[-1],
                    )
                else:
                    logger.debug(
                        "Trial %s is completed, queuing it to attempt forking.", trial
                    )
                    self._queue.append(trial)

    def observe(self, trials: list[Trial]):
        """Observe the trials and queue those available for promotion or forking.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        trials_to_verify = self._triage(trials)
        self._queue_trials_for_promotions(trials_to_verify)


class Lineages(Iterable["LineageNode"]):
    """Lineages of trials for workers in PBT

    This class regroup all lineages of trials generated by PBT for a given experiment.

    Each lineage is a path from a leaf trial (highest fidelity level) up to the root
    (lowest fidelity level). Multiple lineages can fork from the same root, forming a tree.
    A Lineages object may reference multiple trees of lineages. Iterating a Lineages object will
    iterate on the roots of these trees.

    """

    def __init__(self):
        self._lineage_roots: list[LineageNode] = []
        self._trial_to_lineages: dict[str, LineageNode] = {}

    def __len__(self) -> int:
        """Number of roots in the Lineages"""
        return len(self._lineage_roots)

    def __iter__(self):
        """Iterate over the roots of the Lineages"""
        return iter(self._lineage_roots)

    def add(self, trial: Trial) -> LineageNode:
        """Add a trial to the lineages

        If the trial is already in the lineages, this will only return the corresponding lineage
        node. Otherwise, a new lineage node will be created and added as a root.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
            Trial from a `orion.algo.space.Space`.

        Returns
        -------
        orion.algo.pbt.pbt.LineageNode
            The lineage node for the given trial.

        """
        if trial.id in self._trial_to_lineages:
            return self._trial_to_lineages[trial.id]

        lineage = LineageNode(trial)
        self._lineage_roots.append(lineage)
        self._trial_to_lineages[trial.id] = lineage
        return lineage

    def fork(self, base_trial: Trial, new_trial: Trial) -> LineageNode:
        """Fork a base trial to a new one.

        The base trial should already be registered in the Lineages

        Parameters
        ----------
        base_trial: ``orion.core.worker.trial.Trial``
            The base trial that will be the parent lineage node.
        new_trial: ``orion.core.worker.trial.Trial``
            The new trial that will be the child lineage node.

        Raises
        ------
        KeyError
            If the base trial is not already registered in the Lineages

        """

        new_lineage = self._trial_to_lineages[base_trial.id].fork(new_trial)
        self._trial_to_lineages[new_trial.id] = new_lineage
        return new_lineage

    def get_lineage(self, trial: Trial) -> LineageNode:
        """Get the lineage node corresponding to a given trial.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
            The trial for which the function should return the corresponding lineage node.

        Raises
        ------
        KeyError
            If the base trial is not already registered in the Lineages
        """
        return self._trial_to_lineages[trial.id]

    def set_jump(self, base_trial: Trial, new_trial: Trial) -> None:
        """Set a jump between two trials

        This jump is set to represent the relation between the base trial and the new trial.
        This means the base trial was dropped during exploit and the new trial is the result
        of a fork from another trial selected during exploit.

        Both trials should already be registered in the Lineages.

        Parameters
        ----------
        base_trial: ``orion.core.worker.trial.Trial``
            The base trial that was dropped.
        new_trial: ``orion.core.worker.trial.Trial``
            The new trial that was forked.

        Raises
        ------
        KeyError
            If the base trial or the new trial are not already registered in the Lineages.

        """
        self.get_lineage(base_trial).set_jump(self.get_lineage(new_trial))

    def register(self, trial: Trial) -> LineageNode:
        """Add or save the trial in the Lineages

        If the trial is not already in the Lineages, it is added as root. Otherwise,
        the corresponding lineage node is updated with given trial object.

        Parameters
        ----------
        trial: ``orion.core.worker.trial.Trial``
            The trial to register.

        """
        if trial.id not in self._trial_to_lineages:
            lineage = self.add(trial)
        else:
            lineage = self.get_lineage(trial)
            lineage.register(trial)

        return lineage

    def get_elites(self, max_depth: int | None = None) -> list[Trial]:
        """Get best trials of each lineage

        Each lineage is a path from a leaf to the root. When there is a forking,
        the path followed is not from child (new trial) to parent (forked trial), but
        rather to base trial (trial dropped). This is to represent the path taken
        by the sequence of trial execution within a worker. This also avoids having
        duplicate elite trials on different lineages.

        Best trials may be looked for up to a ``max_depth``.

        Parameters
        ----------
        max_depth: int or ``orion.core.worker.trial.Trial``, optional
            The maximum depth to look for best trials. It can be an int to represent the depth
            directly, or a trial, from which the depth will be inferred. If a trial, this trial
            should be in the Lineages. Default: None, that is, no max depth.
        """
        if max_depth and not isinstance(max_depth, int):
            max_depth = self.get_lineage(max_depth).node_depth

        def get_parent_at_depth(node, depth):
            while node.node_depth > depth:
                node = node.parent

            return node

        trials: list[Trial] = []
        for lineage in self._lineage_roots:
            nodes = lineage.leafs

            if max_depth is not None:
                trimmed_nodes = set()
                for node in nodes:
                    node = get_parent_at_depth(node, max_depth)
                    trimmed_nodes.add(node)
                nodes = list(trimmed_nodes)

            for node in nodes:
                if node.jumps and (
                    (max_depth is None) or (node.node_depth < max_depth)
                ):
                    continue

                best_trial = node.get_best_trial()
                if best_trial is not None:
                    trials.append(best_trial)

        return trials

    def get_trials_at_depth(self, trial_or_depth: int | Trial) -> list[Trial]:
        """Returns the trials of all lineages at a given depth

        Parameters
        ----------
        trial_or_depth: int or ``orion.core.worker.trial.Trial``
            If an int, this represents the depth directly. If a trial, the depth will be inferred
            from it. This trial should be in the Lineages.

        Raises
        ------
        KeyError
            If depth is inferred from trial but trial is not already registered in the Lineages

        """
        if isinstance(trial_or_depth, int):
            depth = trial_or_depth
        else:
            depth = self.get_lineage(trial_or_depth).node_depth

        trials: list[Trial] = []
        for lineage in self._lineage_roots:
            for trial_node in lineage.get_nodes_at_depth(depth):
                trials.append(trial_node.item)

        return trials


class LineageNode(TreeNode[Trial]):
    """
    Lineage node

    The lineage node is based on :py:class:`orion.core.utils.tree.TreeNode`. It provides
    additional methods to help represent lineages for PBT, in particular, ``fork``,
    ``set_jump``, ``get_true_ancestor`` and ``get_best_trial``.

    A lineage node can be connected to a parent and children, like a typical TreeNode, but
    also to ``jumps`` and a ``base``. The jumps and base represent the connection between nodes
    when PBT drops a trial and rather fork another one. In such case, the dropped trial
    will refer to the new trial (the forked one) with ``jumps`` (it can refer to many if
    the new trials crashed and required rollback) and the forked trial will refer to the
    dropped one with ``base`` (it can only refer one).

    Parameters
    ----------
    trial: ``orion.core.worker.trial.Trial``
        The trial to represent with the lineage node.
    parent: LineageNode, optional
        The parent node for this lineage node. Default: None, that is, no parent.

    """

    def __init__(self, trial: Trial, parent: LineageNode | None = None):
        super().__init__(copy.deepcopy(trial), parent=parent)
        self._jump: TreeNode[LineageNode] = TreeNode(self)

    @property
    def tree_name(self) -> str:
        """Name of the node for pretty printing."""
        return str(self.item)

    @property
    def jumps(self) -> list[LineageNode]:
        """New trials generated from forks when dropping this node."""
        return [node.item for node in self._jump.children]

    @property
    def base(self) -> LineageNode | None:
        """Base trial that was dropped in favor of this forked trial, if this trial resulted from a
        fork.
        """
        return self._jump.parent.item if self._jump.parent else None

    def register(self, trial: Trial) -> None:
        """Save the trial object.

        Register will copy the object so that any modifications on it externally will not
        impact the interval representation of the Lineage node.
        """
        self.item = copy.deepcopy(trial)

    def fork(self, new_trial: Trial) -> LineageNode:
        """Fork the trial to the new one.

        A new lineage node referring to ``new_trial`` will be created and added as a child
        to current node.

        Parameters
        ----------
        new_trial: ``orion.core.worker.trial.Trial``
            A new trial that is a child of the current one.

        Returns
        -------
        LineageNode
            LineageNode referring to ``new_trial``

        """
        return LineageNode(new_trial, parent=self)

    def set_jump(self, node: LineageNode) -> None:
        """Set the jump to given node

        This will also have the effect of setting ``node.base = self``.

        Parameters
        ----------
        node: LineageNode
            Node to refer to as the jump targen for the current node.

        Raises
        ------
        RuntimeError
            If the given node already has a base.

        """
        if node._jump.parent is not None:
            raise RuntimeError(
                "Trying to jump to an existing node. Jumps to another lineage should only "
                "occur on new nodes."
            )

        node._jump.set_parent(self._jump)

    def get_true_ancestor(self) -> LineageNode | None:
        """Return the base if current trial is the result of a fork, otherwise return parent if is
        has one, otherwise returns None."""
        if self.base is not None:
            return self.base

        if self.parent is not None:
            return self.parent

        return None

    def get_best_trial(self) -> Trial | None:
        """Return best trial on the path from root up to this node.

        The path followed is through `true` ancestors, that is, looking at
        base if the current node is the result of a fork, otherwise looking at the parent.

        Only leaf node trials may not be completed. If there is only one node in the tree
        and the node's trial is not completed, ``None`` is returned instead of a trial object.

        Returns
        -------
        ``None``
            Only one node in the tree and it is not completed.

        ``orion.core.worker.trial.Trial``
            Trial with best objective (lowest).

        """
        parent_node = self.get_true_ancestor()

        if parent_node:
            parent_trial = parent_node.get_best_trial()

            if get_objective(parent_trial) <= get_objective(self.item):
                return parent_trial

        if self.item.status != "completed":
            return None

        return self.item
