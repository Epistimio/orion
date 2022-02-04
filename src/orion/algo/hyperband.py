# -*- coding: utf-8 -*-
"""
A Novel Bandit-Based Approach to Hyperparameter Optimization
============================================================

Implement Hyperband to exploit configurations with fixed resource efficiently

"""
import copy
import hashlib
import logging
from collections import OrderedDict

import numpy
from tabulate import tabulate

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Fidelity
from orion.core.utils.flatten import flatten

logger = logging.getLogger(__name__)

REGISTRATION_ERROR = """
Bad fidelity level {fidelity}. Should be in {budgets}.
Params: {params}
"""

SPACE_ERROR = """
Hyperband cannot be used if space does not contain a fidelity dimension.
For more information on the configuration and usage of Hyperband, see
https://orion.readthedocs.io/en/develop/user/algorithms.html#hyperband
"""

BUDGET_ERROR = """
Cannot build budgets below max_resources;
(max: {}) - (min: {}) > (num_rungs: {})
"""


def compute_budgets(max_resources, reduction_factor):
    """Compute the budgets used for each execution of hyperband"""
    num_brackets = int(numpy.log(max_resources) / numpy.log(reduction_factor))
    budgets = []
    budgets_tab = {}  # just for display consideration
    for bracket_id in range(0, num_brackets + 1):
        bracket_budgets = []
        num_trials = int(
            numpy.ceil(
                int((num_brackets + 1) / (num_brackets - bracket_id + 1))
                * (reduction_factor ** (num_brackets - bracket_id))
            )
        )

        min_resources = max_resources / reduction_factor ** (num_brackets - bracket_id)
        for i in range(0, num_brackets - bracket_id + 1):
            n_i = int(num_trials / reduction_factor ** i)
            min_i = int(min_resources * reduction_factor ** i)
            bracket_budgets.append((n_i, min_i))

            if budgets_tab.get(i):
                budgets_tab[i].append((n_i, min_i))
            else:
                budgets_tab[i] = [(n_i, min_i)]

        budgets.append(bracket_budgets)

    display_budgets(budgets_tab, max_resources, reduction_factor)

    return budgets


def tabulate_status(brackets):

    header = ["i"] + ["n_i", "r_i"] * len(brackets)

    data = []

    num_rungs = max(len(bracket.rungs) for bracket in brackets)
    for rung_id in range(num_rungs):
        row = [rung_id]
        for bracket in brackets:
            if len(bracket.rungs) <= rung_id:
                row.extend(["", ""])
                continue
            in_i = len(bracket.rungs[rung_id]["results"])
            n_i = bracket.rungs[rung_id]["n_trials"]
            r_i = bracket.rungs[rung_id]["resources"]
            row.append("{:>3}/{:>3}".format(in_i, n_i))
            row.append(r_i)
        data.append(row)
    table = tabulate(data, header, tablefmt="github")
    logger.info(table)


def display_budgets(budgets_tab, max_resources, reduction_factor):
    """Display hyperband budget as a table in debug log"""
    num_brackets = len(budgets_tab[0])
    table_str = "Display Budgets:\n"
    col_format_str = "{:<4}" + " {:<12}" * num_brackets + "\n"
    col_title_list = ["i  "] + ["n_i  r_i"] * num_brackets
    col_sub_list = ["---"] + ["---------"] * num_brackets
    table_str += col_format_str.format(*col_sub_list)
    table_str += col_format_str.format(*col_title_list)
    table_str += col_format_str.format(*col_sub_list)

    total_trials = 0
    for key, values in sorted(budgets_tab.items()):
        table_row = "{:<4} ".format(key)
        for value in values:
            n_i, r_i = value
            total_trials += n_i
            st = "{:<5} {:<7}".format(n_i, r_i)
            table_row += st
        table_str += table_row + "\n"
    table_str += col_format_str.format(*col_sub_list)
    table_str += "max resource={}, eta={}, trials number of one execution={}\n".format(
        max_resources, reduction_factor, total_trials
    )
    logger.info(table_str)


class Hyperband(BaseAlgorithm):
    """Hyperband formulates hyperparameter optimization as a pure-exploration non-stochastic
    infinite-armed bandit problem where a predefined resource like iterations, data samples,
    or features is allocated to randomly sampled configurations.`

    For more information on the algorithm,
    see original paper at http://jmlr.org/papers/v18/16-558.html.

    Li, Lisha et al. "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
    Journal of Machine Learning Research, 18:1-52, 2018.

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    repetitions: int
        Number of executions for Hyperband. A single execution of Hyperband takes a finite budget of
        ``(log(R)/log(eta) + 1) * (log(R)/log(eta) + 1) * R``, and ``repetitions`` allows you to run
        multiple executions of Hyperband. Default is ``numpy.inf`` which means to run Hyperband
        until no new trials can be suggested.

    """

    def __init__(self, space, seed=None, repetitions=numpy.inf):
        self.brackets = []
        super(Hyperband, self).__init__(space, seed=seed, repetitions=repetitions)

        self.trial_to_brackets = {}  # Stores Point id (with no fidelity) -> Bracket
        self._trials_info = {}

        fidelity_index = self.fidelity_index
        if fidelity_index is None:
            raise RuntimeError(SPACE_ERROR)

        fidelity_dim = space[fidelity_index]

        self.min_resources = fidelity_dim.low
        self.max_resources = fidelity_dim.high
        self.reduction_factor = fidelity_dim.base

        # if self.reduction_factor < 2:
        #     raise AttributeError("Reduction factor for Hyperband needs to be at least 2.")

        self.repetitions = repetitions

        if self.reduction_factor >= 2:
            self.budgets = compute_budgets(self.max_resources, self.reduction_factor)
            self.brackets = self.create_brackets()
        else:
            self.budgets = None
            self.brackets = None
            logger.warning("Reduction factor for Hyperband needs to be at least 2")

        self.seed_rng(seed)

    def create_bracket(self, i, budgets, iteration):
        return HyperbandBracket(self, budgets, iteration)

    def sample_from_bracket(self, bracket, num):
        """Sample new trials from bracket"""
        trials = []
        while len(trials) < num:
            trial = bracket.get_sample()
            if trial is None:
                break

            trial = trial.branch(
                params={self.fidelity_index: bracket.rungs[0]["resources"]}
            )

            full_id = self.get_id(trial, ignore_fidelity=False, ignore_parent=False)
            id_wo_fidelity = self.get_id(
                trial, ignore_fidelity=True, ignore_parent=True
            )

            bracket_id = self.trial_to_brackets.get(id_wo_fidelity, None)
            if bracket_id is not None:
                bracket_observed = self.brackets[bracket_id]
            else:
                bracket_observed = None

            if not self.has_suggested(trial) and (
                not bracket_observed
                or (
                    bracket_observed.repetition_id < bracket.repetition_id
                    and bracket_observed.get_trial_max_resource(trial)
                    < bracket.rungs[0]["resources"]
                )
            ):
                # if no duplicated found or the duplicated found existing in previous hyperband
                # execution with less resource

                trials.append(trial)
                self.register(trial)
                self.trial_to_brackets[id_wo_fidelity] = self.brackets.index(bracket)

        return trials

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.seed = seed
        self.rng = numpy.random.RandomState(seed)
        self.seed_brackets(seed)

    def seed_brackets(self, seed):
        rng = numpy.random.RandomState(seed)
        for i, bracket in enumerate(self.brackets[::-1]):
            bracket.seed_rng(tuple(rng.randint(0, 1000000, size=3)))

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return {
            "rng_state": self.rng.get_state(),
            "seed": self.seed,
            "_trials_info": copy.deepcopy(dict(self._trials_info)),
            "trial_to_brackets": copy.deepcopy(dict(self.trial_to_brackets)),
            "brackets": [bracket.state_dict for bracket in self.brackets],
        }

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.seed_rng(state_dict["seed"])
        self.rng.set_state(state_dict["rng_state"])
        self._trials_info = state_dict["_trials_info"]
        self.trial_to_brackets = state_dict["trial_to_brackets"]

        while len(self.brackets) < len(state_dict["brackets"]):
            self.append_brackets()
        assert len(self.brackets) == len(state_dict["brackets"]), "corrupted state"

        for bracket, bracket_state_dict in zip(self.brackets, state_dict["brackets"]):
            bracket.set_state(bracket_state_dict)

    def register_samples(self, bracket, samples):
        for sample in samples:
            full_id = self.get_id(sample, ignore_fidelity=False, ignore_parent=False)
            if self.has_observed(sample):
                raise RuntimeError(
                    "Hyperband resampling a trial that was already completed. "
                    "This should never happen. "
                    "If you get this error please report this issue on github at "
                    "https://github.com/Epistimio/orion/issues/new/choose"
                )
            self.register(sample)
            bracket.register(sample)

            if (
                self.get_id(sample, ignore_fidelity=True, ignore_parent=True)
                not in self.trial_to_brackets
            ):
                self.trial_to_brackets[
                    self.get_id(sample, ignore_fidelity=True, ignore_parent=True)
                ] = self.brackets.index(bracket)

    def promote(self, num):
        samples = []
        for bracket in reversed(self.brackets):
            if bracket.is_ready() and not bracket.is_done:
                bracket_samples = bracket.promote(num - len(samples))
                self.register_samples(bracket, bracket_samples)
                samples += bracket_samples

        return samples

    def sample(self, num):
        samples = []
        for bracket in reversed(self.brackets):
            if not bracket.is_filled:
                bracket_samples = self.sample_from_bracket(
                    bracket, min(num - len(samples), bracket.remainings)
                )
                self.register_samples(bracket, bracket_samples)
                samples += bracket_samples

        return samples

    def suggest(self, num):
        """Suggest a number of new sets of parameters.

        Sample new points until first rung is filled. Afterwards
        waits for all trials to be completed before promoting trials
        to the next rung.

        Parameters
        ----------
        num: int, optional
            Number of points to suggest. Defaults to 1.

        Returns
        -------
        list of points or None
            A list of lists representing points suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        """
        self._refresh_brackets()

        samples = self.promote(num)

        samples.extend(self.sample(max(num - len(samples), 0)))

        tabulate_status(self.brackets)

        if samples:
            return samples

        # Either all brackets are done or none are ready and algo needs to wait for some trials to
        # complete
        if len(self.trial_to_brackets) >= self.space.cardinality:
            logger.warning(
                "The number of unique trials of bottom rungs exceeds the search space "
                "cardinality %i, Hyperband algorithm exits.",
                self.space.cardinality,
            )
        else:
            logger.debug(
                f"{self.__class__.__name__} cannot suggest new samples and must wait "
                "for trials to complete."
            )

        return []

    @property
    def executed_times(self):
        """Counter for how many times Hyperband been executed"""
        if not self.brackets:
            return 0
        executed_times = self.brackets[-1].repetition_id
        all_brackets_done = all(
            bracket.is_done for bracket in self.brackets[-len(self.budgets) :]
        )
        return executed_times - int(not all_brackets_done)

    def _refresh_brackets(self):
        """Refresh bracket if one hyperband execution is done"""
        if all(bracket.is_done for bracket in self.brackets):
            logger.info(
                "Hyperband execution %i is done, required to execute %s times",
                self.executed_times,
                str(self.repetitions),
            )

            # Continue to the next execution if need
            if self.executed_times < self.repetitions:
                self.append_brackets()

    def append_brackets(self):
        self.brackets = self.brackets + self.create_brackets()
        # Reset brackets seeds
        self.seed_brackets(self.seed)

    def create_brackets(self):
        return [
            self.create_bracket(i, bracket_budgets, self.executed_times + 1)
            for i, bracket_budgets in enumerate(self.budgets)
        ]

    def _get_bracket(self, trial):
        """Get the bracket of a trial"""
        _id_wo_fidelity = self.get_id(trial, ignore_fidelity=True, ignore_parent=True)
        return self.brackets[self.trial_to_brackets[_id_wo_fidelity]]

    def observe(self, trials):
        """Observe evaluation `results` corresponding to list of `trials` in
        space.

        Parameters
        ----------
        trials: list of ``orion.core.worker.trial.Trial``
           Trials from a `orion.algo.space.Space`.

        """
        for trial in trials:

            if not self.has_suggested(trial):
                logger.debug(
                    "Ignoring trial %s because it was not sampled by current algo.",
                    trial,
                )
                continue

            self.register(trial)

            bracket = self._get_bracket(trial)

            try:
                bracket.register(trial)
            except IndexError:
                logger.warning(
                    "Trial registered to wrong bracket. This is likely due "
                    "to a corrupted database, where trials of different fidelity "
                    "have a wrong timestamps."
                )
                continue

    @property
    def is_done(self):
        """Return True, if all required execution been done."""
        if self.executed_times >= self.repetitions:
            return True
        return False


class HyperbandBracket:
    """Bracket of rungs for the algorithm Hyperband.

    Parameters
    ----------
    hyperband: `Hyperband` algorithm
        The hyperband algorithm object which this bracket will be part of.
    budgets: list of tuple
        Each tuple gives the (n_trials, resource_budget) for the respective rung.
    repetition_id: int
        The id of hyperband execution this bracket belongs to

    """

    def __init__(self, hyperband, budgets, repetition_id):
        self.hyperband = hyperband
        self.rungs = [
            dict(resources=budget, n_trials=n_trials, results=OrderedDict())
            for n_trials, budget in budgets
        ]
        self.seed = None
        self.repetition_id = repetition_id
        self.buffer = 10
        self._samples = None

        logger.debug("Bracket budgets: %s", str(budgets))

    @property
    def state_dict(self):
        return {
            "rungs": copy.deepcopy(self.rungs),
            "samples": copy.deepcopy(self._samples),
        }

    def set_state(self, state_dict):
        self.rungs = state_dict["rungs"]
        self._samples = state_dict["samples"]

    @property
    def is_filled(self):
        """Return True if first rung with trials is filled"""
        return self.has_rung_filled(0)

    def get_trial_max_resource(self, trial):
        """Return the max resource value that has been tried for a trial"""
        max_resource = 0
        _id_wo_fidelity = self.hyperband.get_id(
            trial, ignore_fidelity=True, ignore_parent=True
        )
        for rung in self.rungs:
            if _id_wo_fidelity in rung["results"]:
                max_resource = rung["resources"]

        return max_resource

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self._samples = None

    def get_sample(self):
        if self._samples is None:
            n_samples = int(self.rungs[0]["n_trials"] * self.buffer)
            self._samples = self.hyperband.space.sample(n_samples, seed=self.seed)

        return self._samples.pop(0) if self._samples else None

    def sample(self, num):
        """Sample a new trial with lowest fidelity"""
        should_have_n_trials = self.rungs[0]["n_trials"]
        n_trials = len(self.rungs[0]["results"])
        request = max(min(should_have_n_trials - n_trials, num), 0)
        if request == 0:
            return []

        return self.hyperband.sample_for_bracket(
            request, self, buffer=should_have_n_trials * 10 / request
        )

    def register(self, trial):
        """Register a trial in the corresponding rung"""
        self._get_results(trial)[
            self.hyperband.get_id(trial, ignore_fidelity=True, ignore_parent=True)
        ] = (
            trial.objective.value if trial.objective else None,
            copy.deepcopy(trial),
        )

    def _get_results(self, trial):
        fidelity = flatten(trial.params)[self.hyperband.fidelity_index]
        rungs = [
            rung["results"] for rung in self.rungs if rung["resources"] == fidelity
        ]
        if not rungs:
            budgets = [rung["resources"] for rung in self.rungs]
            raise IndexError(
                REGISTRATION_ERROR.format(
                    fidelity=fidelity, budgets=budgets, params=trial.params
                )
            )

        return rungs[0]

    @property
    def remainings(self):
        should_have_n_trials = self.rungs[0]["n_trials"]
        have_n_trials = len(self.rungs[0]["results"])
        return max(should_have_n_trials - have_n_trials, 0)

    def get_candidates(self, rung_id):
        """Get a candidate for promotion

        Raises
        ------
        TypeError
            If get_candidates is called before the entire rung is completed.
        """
        if self.has_rung_filled(rung_id + 1):
            return []

        rung = self.rungs[rung_id]["results"]
        next_rung = self.rungs[rung_id + 1]["results"]

        rung = sorted(rung.values(), key=lambda pair: pair[0])

        if not rung:
            return []

        should_have_n_trials = self.rungs[rung_id + 1]["n_trials"]
        trials = []
        i = 0
        while len(trials) + len(next_rung) < should_have_n_trials:
            objective, trial = rung[i]
            assert objective is not None
            _id = self.hyperband.get_id(trial, ignore_fidelity=True, ignore_parent=True)
            if _id not in next_rung:
                trials.append(trial)
            i += 1

        return trials

    @property
    def is_done(self):
        """Return True, if the last rung is filled."""
        return self.has_rung_filled(len(self.rungs) - 1)

    def has_rung_filled(self, rung_id):
        """Return True, if the rung[rung_id] is filled."""
        n_trials = len(self.rungs[rung_id]["results"])
        return n_trials >= self.rungs[rung_id]["n_trials"]

    def is_ready(self, rung_id=None):
        """Return True, if the bracket is ready for next promote"""
        if rung_id is not None:
            return self.has_rung_filled(rung_id) and all(
                objective is not None
                for objective, _ in self.rungs[rung_id]["results"].values()
            )

        is_ready = False
        for _rung_id in range(len(self.rungs)):
            if self.has_rung_filled(_rung_id):
                is_ready = self.is_ready(_rung_id)
            else:
                break

        return is_ready

    def promote(self, num):
        """Promote the first candidate that is found and return it

        The rungs are iterated over in reversed order, so that high rungs
        are prioritised for promotions. When a candidate is promoted, the loop is broken and
        the method returns the promoted trial.

        .. note ::

            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.

        """
        if self.is_done:
            return []

        for rung_id in range(len(self.rungs)):
            # No more promotion possible, skip to next rung
            if self.has_rung_filled(rung_id + 1):
                continue

            if not self.is_ready(rung_id):
                return []

            trials = []
            for candidate in self.get_candidates(rung_id):
                # pylint: disable=logging-format-interpolation
                logger.debug(
                    "Promoting {trial} from rung {past_rung} with fidelity {past_fidelity} to "
                    "rung {new_rung} with fidelity {new_fidelity}".format(
                        trial=candidate,
                        past_rung=rung_id,
                        past_fidelity=flatten(candidate.params)[
                            self.hyperband.fidelity_index
                        ],
                        new_rung=rung_id + 1,
                        new_fidelity=self.rungs[rung_id + 1]["resources"],
                    )
                )

                candidate = candidate.branch(
                    status="new",
                    params={
                        self.hyperband.fidelity_index: self.rungs[rung_id + 1][
                            "resources"
                        ]
                    },
                )
                if not self.hyperband.has_suggested(candidate):
                    trials.append(candidate)

            return trials[:num]

        return []

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.seed = seed

    def __repr__(self):
        """Return representation of bracket with fidelity levels"""
        return "{}(resource={}, repetition id={})".format(
            self.__class__.__name__,
            [rung["resources"] for rung in self.rungs],
            self.repetition_id,
        )
