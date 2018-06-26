# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.analysers.lpi` -- LPI analyser
==============================================

.. module:: lpi
   :platform: Unix
   :synopsis:

"""

import logging

import numpy as np

from orion.viz.analysers import BaseAnalyser
from orion.viz.analysis import TimeSeriesAnalysis

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, \
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

log = logging.getLogger(__name__)


class LPI(BaseAnalyser):

    def __init__(self, trials, experiment, regressor_name, target_name, target_args,
                 n_trials_bootstraps, **regressor_args):
        self.target_args = []
        self.n_trials_bootstraps = 1
        super(LPI, self).__init__(trials, experiment, regressor_name=regressor_name,
                                  target_name=target_name, target_args=target_args,
                                  n_trials_bootstraps=n_trials_bootstraps,
                                  regressor_args=regressor_args)

        self._regressors_ = {
            'AdaBoostRegressor': AdaBoostRegressor,
            'BaggingRegressor': BaggingRegressor,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'RandomForestRegressor': RandomForestRegressor,
        }

        self._targets_ = {
            'error_rate': self.retrieve_error_rate,
            'trial_time': self.retrieve_trial_time,
        }

        self.n_trials_bootstraps = float(self.n_trials_bootstraps)

    def retrieve_params(self):
        # Get a list of the name of all the parameters
        params_key = list(self.space)
        return params_key, [[param.value for param in trial.params] for trial in self.trials]

    def retrieve_targets(self):
        if self.target_name not in self._targets_.keys():
            raise KeyError('%s is not a supported target. Did you mean any of theses: %s' %
                           (self.target_name, ','.join(list(self._targets_.keys()))))

        return self._targets_[self.target_name](*self.target_args)

    def retrieve_error_rate(self, key):
        return [result.value for trial in self.trials for result in trial.results
                if result.name == key]

    def retrieve_trial_time(self):
        return [(trial.end_time - trial.start_time).total_seconds() for trial in self.trials]

    def train_epm(self, params, target):
        if self.regressor_name not in self._regressors_:
            raise KeyError('%s is not a supported regressor. Did you mean any of theses: \%s' %
                           (self.regressor_name, ','.join(list(self._regressors_.keys()))))

        # TODO fix hack here. See other comment under function transform_categorical
        hacked_data = [[self.hack(param) for param in row] for row in params]
        regressor = self._regressors_[self.regressor_name](**self.regressor_args)
        return regressor.fit(hacked_data, target)

    def hack(self, param):
        if type(param).__name__ != 'str' and \
           type(param).__name__ != 'str_':  # numpy thing. TODO fix this
            return param
        return param == 'NONE'

    def compute_param_grid(self, trial_params):
        n_trials = min(len(trial_params), self.n_trials_bootstraps)
        epm_trials_idx = np.random.choice(len(trial_params), n_trials, replace=False)
        epm_trials = np.array(trial_params)[epm_trials_idx]
        unique_params = [set(trial) for trial in np.transpose(trial_params, [1, 0])]
        return [[self.generate_epm_trial(epm_trials, idx, param) for param in params]
                for idx, params in enumerate(unique_params)]

    def generate_epm_trial(self, epm_trials, idx, param):
        trials = np.copy(epm_trials)
        for trial in trials:
            trial[idx] = param
        return trials

    def transform_categorical(self, params_grid):
        # TODO: This is currently an ugly hack to make thing work.
        # A real solution would require a deeper analysis of the
        # params_grid or a different architecture.
        # This will have to be fixed after the submission.
        return [[[[self.hack(param) for param in row] for row in param_rows]
                for param_rows in params] for params in params_grid]

    def compute_scores(self, epm, param_grid):
        return [[np.mean(epm.predict(params)) for params in params_rows]
                for params_rows in param_grid]

    def lpi(self, scores):
        margin = scores.sum()
        return scores / margin

    def analyse(self, of_type=None):
        params_key, params = self.retrieve_params()
        target = self.retrieve_targets()
        epm = self.train_epm(params, target)
        params_grid = self.compute_param_grid(params)
        params_grid = self.transform_categorical(params_grid)
        scores = self.compute_scores(epm, params_grid)
        var_scores = np.array([np.var(score) for score in scores])
        return TimeSeriesAnalysis(dict(zip(params_key, self.lpi(var_scores))))

    @property
    def available_analysis(self):
        return [TimeSeriesAnalysis]
