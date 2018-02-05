# -*- coding: utf-8 -*-
"""
:mod:`metaopt.core.worker.consumer` -- Evaluate objective on a set of parameters
================================================================================

.. module:: consumer
   :platform: Unix
   :synopsis: Call user's script as a black box process to evaluate a trial.

"""
import logging
import os
import subprocess
import tempfile

import six

from metaopt.core.io.convert import JSONConverter
from metaopt.core.io.database import Database
from metaopt.core.io.space_builder import SpaceBuilder
from metaopt.core.utils import SingletonType
from metaopt.core.worker.trial import Trial

log = logging.getLogger(__name__)


@six.add_metaclass(SingletonType)
class Consumer(object):
    """Consume a trial by using it to initialize a black-box box to evaluate it.

    It uses an `Experiment` object to push an evaluated trial, if results are
    delivered to the worker process successfully.

    It forks another process which executes user's script with the suggested
    options. It expects results to be written in a file, whose path has been
    defined in a special metaopt environmental variable which is set into the
    child process' environment.

    """

    def __init__(self, experiment):
        """Initialize a consumer.

        :param experiment: Manager of this experiment, provides convenient
           interface for interacting with the database.
        """
        self.experiment = experiment
        self.space = experiment.space
        if self.space is None:
            raise RuntimeError("Experiment object provided to Consumer has not yet completed"
                               " initialization.")

        # Fetch space builder
        self.template = SpaceBuilder()
        # Get path to user's script and infer trial configuration directory
        self.script = experiment.metadata['user_script']
        self.tmp_dir = os.path.join(tempfile.gettempdir(), 'metaopt')
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.converter = JSONConverter()

    def consume(self, trial):
        """Execute user's script as a block box using the options contained
        within `trial`.

        :type trial: `metaopt.core.worker.trial.Trial`

        """
        with tempfile.TemporaryDirectory(prefix=self.experiment.name + '_',
                                         dir=self.tmp_dir) as workdirname:
            completed_trial = self._consume(trial, workdirname)

        if completed_trial is not None:
            self.experiment.push_completed_trial(completed_trial)
        else:
            trial.status = 'new'  # recycle failed trial
            Database().write('trials', trial.to_dict(),
                             query={'_id': trial.id})

    def _consume(self, trial, workdirname):
        config_file = tempfile.NamedTemporaryFile(mode='w', prefix='trial_',
                                                  suffix='.conf', dir=workdirname,
                                                  delete=False)
        config_file.close()
        results_file = tempfile.NamedTemporaryFile(mode='w', prefix='results_',
                                                   suffix='.log', dir=workdirname,
                                                   delete=False)
        results_file.close()

        cmd_args = self.template.build_to(config_file.name, trial)
        script_process = self.launch_process(results_file.name, cmd_args)

        if script_process is None:
            return None

        returncode = script_process.wait()

        if returncode != 0:
            log.error("Something went wrong. Check logs. Process "
                      "returned with code %d !", returncode)
            return None

        results = self.converter.parse(results_file.name)

        trial.results = [Trial.Result(name=res['name'],
                                      type=res['type'],
                                      value=res['value']) for res in results]

        return trial

    def launch_process(self, results_filename, cmd_args):
        """Facilitate launching a black-box trial."""
        env = dict(os.environ)
        env['METAOPT_RESULTS_PATH'] = str(results_filename)
        command = [self.script] + cmd_args
        process = subprocess.Popen(command, env=env)
        returncode = process.poll()
        if returncode is not None and returncode < 0:
            log.error("Failed to execute script to evaluate trial. Process "
                      "returned with code %d !", returncode)
            return None

        return process
