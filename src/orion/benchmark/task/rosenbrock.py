from collections import defaultdict

import numpy

from orion.storage.base import setup_storage
from orion.benchmark.base import BaseTask, BaseAssess
from orion.client import create_experiment


def rosenbrock(x):
    """Evaluate a n-D rosenbrock function."""
    x = numpy.asarray(x)
    summands = 100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2
    y = numpy.sum(summands)
    return [dict(
        name='rosenbrock',
        type='objective',
        value=y)]


class RosenBrock(BaseTask):

    # assessments that the particular task supports
    assessments = {
        'TimeToResult': {
            'space': {
                'x': 'uniform(1, 3, shape=2)'
            },
            'max_trials': 10,
            'strategy': None,
        },
        'AverageResult': {
            'space': {
                'x': 'uniform(1, 3, shape=2)'
            },
            'max_trials': 10,
            'strategy': None,
        },
        'AverageRank': {
            'space': {
                'x': 'uniform(1, 3, shape=2)'
            },
            'max_trials': 10,
            'strategy': None,
        },
    }

    def __init__(self, name, algorithm, assess):
        """
        - build orion experiment
        """

        # TODO: Should use the same storage configure for all experiments in the same benchmark
        setup_storage(storage=None, debug=False)

        if isinstance(assess, BaseAssess):
            assess_key = type(assess).__name__
        else:
            assess_key = assess
        assessments = self.assessments[assess_key]

        self.experiment = create_experiment(
            name, space=assessments['space'], algorithms=algorithm,
            max_trials=assessments['max_trials'], strategy=assessments['strategy'])

        self.name = name
        self.algorithm = algorithm

    def run(self):
        """
        - run the orion experiment
        :return:
        """
        return self.experiment.workon(rosenbrock, self.experiment.max_trials)

    def status(self):
        """
        - status of the orion experiment
        {
          'algorithm': 'random',
          'task': 'testbench1',
          'experiments': [
            {
              'experiment': 'testbench11',
              'is_done': True,
              'trials': {
                'completed': [
                  Trial(experiment=ObjectId('5f11b152008efcfa1015d773'), status='completed', params=/x:[2.615, 1.71]),
                  Trial(experiment=ObjectId('5f11b152008efcfa1015d773'), status='completed', params=/x:[1.002, 2.004]),
                  ...
                ]
              }
            }
          ]
        }
        """
        task_status = {'task': self.name, 'algorithm': self.algorithm, 'experiments': []}
        exp_status = {'experiment': self.name}

        trials = self.experiment.fetch_trials()
        status_dict = defaultdict(list)
        for trial in trials:
            status_dict[trial.status].append(trial)
        exp_status['trials'] = status_dict
        exp_status['is_done'] = self.experiment.is_done

        task_status['experiments'].append(exp_status)

        return task_status

    def performance(self):
        """
        - formatted the experiment result for the particular assess
        :return:
        """
        # NOTE(Xavier): What is this method for?
        # NOTE(Lin): It will be used by the assessment to show the task result on one algorithm
        return [self.experiment]
