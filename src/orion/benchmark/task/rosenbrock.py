import os

import orion.core
import orion.core.io.experiment_builder as experiment_builder
from orion.storage.base import setup_storage

from orion.core.io import resolve_config
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser
from orion.core.worker import workon

from orion.benchmark.base import BaseTask, BaseAssess


class RosenBrock(BaseTask):

    # assessments that the particular task supports
    assessments = {
        'TimeToResult': {
            'user_args': ['python', 'scripts/rosenbrock.py', '--x~uniform(1,3, shape=(2))'],
            'max_trails': 10,
            'strategy': None,
        },
        'AverageResult': {
            'user_args': ['python', 'scripts/rosenbrock.py', '--x~uniform(1,3, shape=(2))'],
            'max_trails': 10,
            'strategy': None,
        }
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

        user_args = assessments['user_args']
        user_args[1] = os.path.join(os.path.dirname(os.path.abspath(__file__)), user_args[1])

        metadata = resolve_config.fetch_metadata(user_args=user_args)
        parser = OrionCmdlineParser(orion.core.config.worker.user_script_config,
                                    allow_non_existing_user_script=True)
        parser.parse(user_args)
        metadata["parser"] = parser.get_state_dict()
        metadata["priors"] = dict(parser.priors)

        space = metadata["priors"]

        self.experiment = experiment_builder.build(
            name, space=space, algorithms=algorithm, max_trials=assessments['max_trails'],
            strategy=assessments['strategy'], user_args=user_args, metadata=metadata)

        self.name = name
        self.algorithm = algorithm

    def run(self):
        """
        - run the orion experiment
        :return:
        """
        worker_config = orion.core.config.worker.to_dict()
        worker_config['silent'] = True
        workon(self.experiment, **worker_config)

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

        # experiment = experiment_builder.build_view(self.name)

        db_config = experiment_builder.fetch_config_from_db(self.name)
        db_config.setdefault('version', 1)

        experiment = experiment_builder.create_experiment(**db_config)

        trials = experiment.fetch_trials()
        status_dict = {}
        for trial in trials:
            if status_dict.get(trial.status, None):
                status_dict[trial.status].append(trial)
            else:
                status_dict[trial.status] = [trial]
        exp_status['trials'] = status_dict
        exp_status['is_done'] = experiment.is_done

        task_status['experiments'].append(exp_status)

        return task_status

    def performance(self):
        """
        - formatted the experiment result for the particular assess
        :return:
        """
        db_config = experiment_builder.fetch_config_from_db(self.name)
        db_config.setdefault('version', 1)

        experiment = experiment_builder.create_experiment(**db_config)
        return [experiment]