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

import orion.core
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser
from orion.core.utils.working_dir import WorkingDir
from orion.core.worker.trial_pacemaker import TrialPacemaker

log = logging.getLogger(__name__)


# pylint: disable = unused-argument
def _handler(signum, frame):
    log.error('Oríon has been interrupted.')
    raise KeyboardInterrupt


class ExecutionError(Exception):
    """Error raised when Orion is unable to execute the user's script without errors."""

    pass


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
        self.template_builder = OrionCmdlineParser(orion.core.config.user_script_config)
        self.template_builder.set_state_dict(experiment.metadata['parser'])
        # Get path to user's script and infer trial configuration directory
        if experiment.working_dir:
            self.working_dir = os.path.abspath(experiment.working_dir)
        else:
            self.working_dir = os.path.join(tempfile.gettempdir(), 'orion')

        self.script_path = experiment.metadata['user_script']

        self.pacemaker = None

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

                results_file = self._consume(trial, workdirname)

                log.debug("## Parse results from file and fill corresponding Trial object.")
                self.experiment.update_completed_trial(trial, results_file)

        except KeyboardInterrupt:
            log.debug("### Save %s as interrupted.", trial)
            self.experiment.set_trial_status(trial, status='interrupted')
            raise

        except ExecutionError:
            log.debug("### Save %s as broken.", trial)
            self.experiment.set_trial_status(trial, status='broken')

    def get_execution_environment(self, trial, results_file='results.log'):
        """Set a few environment variables to allow users and
        underlying processes to know if they are running under orion.

        Parameters
        ----------
        results_file: str
           file used to store results, this is only used by the legacy protocol
        trial: Trial
           reference to the trial object that is going to be run

        Notes
        -----
        This function defines the environment variables described below

        .. envvar:: ORION_EXPERIMENT_ID

           Current experiment that is being ran.

        .. envvar::  ORION_EXPERIMENT_NAME

           Name of the experiment the worker is currently working on.

        .. envvar::  ORION_EXPERIMENT_VERSION

           Version of the experiment the worker is currently working on.

        .. envvar:: ORION_TRIAL_ID

           Current trial id that is currently being executed in this process.

        .. envvar:: ORION_WORKING_DIRECTORY

           Trial's current working directory.

        .. envvar:: ORION_RESULTS_PATH

           Trial's results file that is read by the legacy protocol to get the results of the trial
           after a successful run.

        """
        env = dict(os.environ)

        env['ORION_EXPERIMENT_ID'] = str(self.experiment.id)
        env['ORION_EXPERIMENT_NAME'] = str(self.experiment.name)
        env['ORION_EXPERIMENT_VERSION'] = str(self.experiment.version)
        env['ORION_TRIAL_ID'] = str(trial.id)

        env['ORION_WORKING_DIR'] = str(trial.working_dir)
        env['ORION_RESULTS_PATH'] = str(results_file)

        return env

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
        env = self.get_execution_environment(trial, results_file.name)
        cmd_args = self.template_builder.format(config_file.name, trial, self.experiment)

        log.debug("## Launch user's script as a subprocess and wait for finish.")

        self.pacemaker = TrialPacemaker(trial)
        self.pacemaker.start()
        try:
            self.execute_process(cmd_args, env)
        finally:
            # merciless
            self.pacemaker.stop()

        return results_file

    def execute_process(self, cmd_args, environ):
        """Facilitate launching a black-box trial."""
        command = [self.script_path] + cmd_args

        signal.signal(signal.SIGTERM, _handler)
        process = subprocess.Popen(command, env=environ)

        return_code = process.wait()
        if return_code != 0:
            raise ExecutionError("Something went wrong. Check logs. Process "
                                 "returned with code {} !".format(return_code))
