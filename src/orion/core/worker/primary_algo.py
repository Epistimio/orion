# -*- coding: utf-8 -*-
"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
import orion.core.utils.backward as backward
from orion.core.worker.transformer import build_required_space


# pylint: disable=too-many-public-methods
class SpaceTransformAlgoWrapper:
    """Perform checks on points and transformations. Wrap the primary algorithm.

    1. Checks requirements on the parameter space from algorithms and create the
    appropriate transformations. Apply transformations before and after methods
    of the primary algorithm.
    2. Checks whether incoming and outcoming points are compliant with a space.

    Parameters
    ----------
    algo_constructor: Child class of `BaseAlgorithm`
        Class constructor to build the algorithm object.
    space : `orion.algo.space.Space`
       The original definition of a problem's parameters space.
    algorithm_config : dict
       Configuration for the algorithm.

    """

    def __init__(self, algo_constructor, space, **algorithm_config):
        self._space = space
        requirements = backward.get_algo_requirements(algo_constructor)
        self.transformed_space = build_required_space(space, **requirements)
        self.algorithm = algo_constructor(space, **algorithm_config)
        self.algorithm.space = self.transformed_space

    def seed_rng(self, seed):
        """Seed the state of the algorithm's random number generator."""
        self.algorithm.seed_rng(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return self.algorithm.state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.algorithm.set_state(state_dict)

    def suggest(self, num):
        """Suggest a `num` of new sets of parameters.

        Parameters
        ----------
        num: int
            Number of points to suggest. The algorithm may return less than the number of points
            requested.

        Returns
        -------
        list of trials or None
            A list of trials representing values suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """
        transformed_trials = self.algorithm.suggest(num)

        if transformed_trials is None:
            return None

        for trial in transformed_trials:
            self._verify_trial(trial, space=self.transformed_space)
            if trial not in self.transformed_space:
                raise ValueError(
                    f"Trial {trial.id} not contained in space:"
                    f"\nParams: {trial.params}\n: Space{self.transformed_space}"
                )

        trials = []
        for transformed_trial in transformed_trials:
            trial = self.transformed_space.reverse(transformed_trial)
            self._verify_trial(trial, space=self.space)
            trials.append(trial)

        return trials

    def observe(self, trials):
        """Observe evaluated trials.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        transformed_trials = []
        for trial in trials:
            self._verify_trial(trial, space=self.space)
            transformed_trials.append(self.transformed_space.transform(trial))
        self.algorithm.observe(transformed_trials)

    def has_suggested(self, trial):
        """Whether the algorithm has suggested a given trial.

        .. seealso:: `orion.algo.base.BaseAlgorithm.has_suggested`
        """
        return self.algorithm.has_suggested(self.transformed_space.transform(trial))

    def has_observed(self, trial):
        """Whether the algorithm has observed a given trial.

        .. seealso:: `orion.algo.base.BaseAlgorithm.has_observed`

        """
        return self.algorithm.has_observed(self.transformed_space.transform(trial))

    @property
    def n_suggested(self):
        """Number of trials suggested by the algorithm"""
        return self.algorithm.n_suggested

    @property
    def n_observed(self):
        """Number of completed trials observed by the algorithm"""
        return self.algorithm.n_observed

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        return self.algorithm.is_done

    def score(self, trial):
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score any parameter (no preference).
        """
        self._verify_trial(trial)
        return self.algorithm.score(self.transformed_space.transform(trial))

    def judge(self, trial, measurements):
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        self._verify_trial(trial)
        return self.algorithm.judge(
            self.transformed_space.transform(trial), measurements
        )

    def should_suspend(self, trial):
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        self._verify_trial(trial)
        return self.algorithm.should_suspend(trial)

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.
        """
        return self.algorithm.configuration

    @property
    def space(self):
        """Domain of problem associated with this algorithm's instance.

        .. note:: Redefining property here without setter, denies base class' setter.
        """
        return self._space

    def get_id(self, point, ignore_fidelity=False):
        """Compute a unique hash for a point based on params"""
        return self.algorithm.get_id(
            self.transformed_space.transform(point), ignore_fidelity=ignore_fidelity
        )

    @property
    def fidelity_index(self):
        """Compute the index of the point where fidelity is.

        Returns None if there is no fidelity dimension.
        """
        return self.algorithm.fidelity_index

    def _verify_trial(self, trial, space=None):
        if space is None:
            space = self.space

        if trial not in space:
            raise ValueError(
                f"Trial {trial.id} not contained in space:"
                f"\nParams: {trial.params}\nSpace: {space}"
            )
