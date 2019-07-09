# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.consumer` -- Evaluate objective on a set of parameters
==============================================================================

.. module:: consumer
   :platform: Unix
   :synopsis: Call user's script as a black box process to evaluate a trial.

"""
import logging
import os
import signal
import subprocess
import tempfile

from orion.core.io.convert import JSONConverter
from orion.core.io.database import Database
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.working_dir import WorkingDir
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)


# pylint: disable = unused-argument
def _handler(signum, frame):
    log.error('Oríon has been interrupted.')
    raise KeyboardInterrupt


class Consumer(object):
    """Consume a trial by using it to initialize a black-box box to evaluate it.

    It uses an `Experiment` object to push an evaluated trial, if results are
    delivered to the worker process successfully.

    It forks another process which executes user's script with the suggested
    options. It expects results to be written in a **JSON** file, whose path
    has been defined in a special orion environmental variable which is set
    into the child process' environment.

    """

    def __init__(self, experiment):
        """Initialize a consumer.

        :param experiment: Manager of this experiment, provides convenient
           interface for interacting with the database.
        """
        log.debug("Creating Consumer object.")
        self.experiment = experiment
        self.space = experiment.space
        if self.space is None:
            raise RuntimeError("Experiment object provided to Consumer has not yet completed"
                               " initialization.")

        # Fetch space builder
        self.template_builder = SpaceBuilder()
        self.template_builder.build_from(experiment.metadata['user_args'])
        # Get path to user's script and infer trial configuration directory
        if experiment.working_dir:
            self.working_dir = os.path.abspath(experiment.working_dir)
        else:
            self.working_dir = os.path.join(tempfile.gettempdir(), 'orion')

        self.script_path = experiment.metadata['user_script']

        self.converter = JSONConverter()

    def consume(self, trial):
        """Execute user's script as a block box using the options contained
        within `trial`.

        :type trial: `orion.core.worker.trial.Trial`

        """
        log.debug("### Create new directory at '%s':", self.working_dir)
        temp_dir = self.experiment.working_dir is None
        prefix = self.experiment.name + "_"
        suffix = trial.id

        try:
            with WorkingDir(self.working_dir, temp_dir,
                            prefix=prefix, suffix=suffix) as workdirname:
                log.debug("## New consumer context: %s", workdirname)
                trial.working_dir = workdirname
                self._consume(trial, workdirname)
        except KeyboardInterrupt:
            log.debug("### Save %s as interrupted.", trial)
            trial.status = 'interrupted'
            Database().write('trials', trial.to_dict(),
                             query={'_id': trial.id})
            raise
        except RuntimeError:
            log.debug("### Save %s as broken.", trial)
            trial.status = 'broken'
            Database().write('trials', trial.to_dict(),
                             query={'_id': trial.id})
            raise
        else:
            log.debug("### Register successfully evaluated %s.", trial)
            self.experiment.push_completed_trial(trial)

    def _consume(self, trial, workdirname):
        config_file = tempfile.NamedTemporaryFile(mode='w', prefix='trial_',
                                                  suffix='.conf', dir=workdirname,
                                                  delete=False)
        config_file.close()
        log.debug("## New temp config file: %s", config_file.name)
        results_file = tempfile.NamedTemporaryFile(mode='w', prefix='results_',
                                                   suffix='.log', dir=workdirname,
                                                   delete=False)
        results_file.close()
        log.debug("## New temp results file: %s", results_file.name)

        log.debug("## Building command line argument and configuration for trial.")
        cmd_args = self.template_builder.build_to(config_file.name, trial, self.experiment)

        log.debug("## Launch user's script as a subprocess and wait for finish.")

        self.execute_process(results_file.name, cmd_args)

        log.debug("## Parse results from file and fill corresponding Trial object.")
        results = self.converter.parse(results_file.name)

        trial.results = [Trial.Result(name=res['name'],
                                      type=res['type'],
                                      value=res['value']) for res in results]

    def execute_process(self, results_filename, cmd_args):
        """Facilitate launching a black-box trial."""
        env = dict(os.environ)
        env['ORION_RESULTS_PATH'] = str(results_filename)
        command = [self.script_path] + cmd_args

        signal.signal(signal.SIGTERM, _handler)
        process = subprocess.Popen(command, env=env)

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError("Something went wrong. Check logs. Process "
                               "returned with code {} !".format(return_code))
