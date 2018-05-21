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
import subprocess
import sys
import tempfile

from orion.core.io.convert import JSONConverter
from orion.core.io.database import Database
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)


class Consumer(object):
    """Consume a trial by using it to initialize a black box to evaluate it.

    It uses an `Experiment` object to push an evaluated trial, if results are
    delivered to the worker process successfully.

    It forks another process which executes user's script with the suggested
    options. It expects results to be written in a **JSON** file, whose path
    has been defined in a special orion environmental variable which is set
    into the child process' environment.

    Attributes
    ----------
    experiment : `orion.core.worker.experiment.Experiment`
       Manager of current experiment
    space : `orion.algo.space.Space`
       Definition of problem's parameter space
    template_builder : `orion.core.io.space_builder.SpaceBuilder`
       Object that will build particular instances of the command line arguments
       and possibly configuration files, corresponding to a particular trial.
    script_path : str
       Path or name of the executable initializing user's code to be optimized
    tmp_dir : str
       Path to base temporary directory in user's system to output instances
       of configuration files, logs and comminucation files for a particular
       trial
    converter : `orion.core.io.converter.JSONConverter`
       Convenience object that parses and generates JSON files
    current_trial : `orion.core.worker.trial.Trial`
       If it is not None, then this is the trial which is being currently
       evaluated by the worker.

    """

    class SuspendTrial(Exception):
        """Raise this to communicate that `self.current_trial`'s evaluation
        has not been completed and that the execution of user's script has been
        suspended.
        """

        pass

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
        # Get path to user's script and infer trial configuration directory
        self.script_path = experiment.metadata['user_script']
        self.tmp_dir = os.path.join(tempfile.gettempdir(), 'orion')
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.converter = JSONConverter()

        self.current_trial = None

    def consume(self, trial):
        """Execute user's script as a block box using the options contained
        within `trial`.

        This function is going to update a `trial` status as *'broken'*
        if user's script fatally crashed during execution, and as *'interrupted'*
        if a catchable terminating os signal was captured.

        It consists the main entry point to the functionality of this object.
        It will be called by `orion.core.worker.workon` to evaluate the
        performance of a particular `trial` on user's script.

        When a `trial` is successfully evaluated, its entry in the database
        is going to be updated with the results reported from user's code
        (described in ``self.current_trial.results``), and a ``'done'`` status.

        :type trial: `orion.core.worker.trial.Trial`

        .. note:: Out of the possible reasons that a user's script may crash,
           three case categories need to be taken into consideration:

           1. **There is a bug in user's code**: Conditional or not, due to a
              syntax or logical error, the process executing user's code will crash
              with a non-predictable code. A trial that was used will be saved in
              the database as *'broken'*.
           2. **Inputs defined by the trial cause an arithmetic NaN**: Ideally
              these errors should be handled by user's code, by catching existent
              NaN arising in their computations, and reporting a result with
              a use-case-specific undesired score. This will help some algorithms
              determine that trials that cause this behaviour are to be avoided.
              This is left as user's responsibility, because reporting an
              arbitrary score may not be compatible with the  use-case-specific
              definition of a trial's objective value and it also violates the
              desired separation of responsibility. If this case is left
              untreated, then (1) holds.
           3. **Inputs provided to user's code are invalid**: User's parsing
              code is expected to fail, if an incompatible set of intpus is given
              to the script. This fatal case should arise when the script's
              parameter space definition does not correspond to the version
              of the actual user's code to be executed. Treatment is left to
              the user. However **a fail fast solution could be detected**,
              because **argparse** (for instance) exits with code 2,
              when such a case happens. This argparse specific treatment
              is disputable. In favour of this handling is that this practice
              is traditional, but still, not enforced.

           .. seealso::

              Method `Consumer.interact_with_script`
                 Code which would exit with 2, if user's script exited with 2.

              `GNU bash manual, Exit Status <Exit Status>`_
                 Quoting: *... return an exit status of 2 to indicate
                 incorrect usage, generally invalid options or missing arguments.*

        .. _Exit Status:
           https://www.gnu.org/software/bash/manual/html_node/Exit-Status.html

        """
        try:
            self.current_trial = trial
            returncode = None
            returncode = self._consume()

        except KeyboardInterrupt:
            new_status = 'interrupted'
            raise

        except SystemExit:
            new_status = 'broken'
            Database().write('experiments', {'status': 'broken'},
                             {'_id': self.experiment._id})  # pylint:disable=protected-access
            raise

        except Consumer.SuspendTrial:
            new_status = 'suspended'

        finally:
            if returncode == 0 or (returncode is None and self.current_trial.results):
                log.debug("### Update successfully evaluated %s.", self.current_trial)
                self.experiment.push_completed_trial(self.current_trial)
            elif returncode is not None:
                self.experiment.push_completed_trial(self.current_trial, 'broken')
            else:
                self.experiment.push_completed_trial(self.current_trial, new_status)
            self.current_trial = None

    def _consume(self):
        log.debug("### Create new temporary directory at '%s':", self.tmp_dir)
        # XXX: wrap up with try/finally and safe-release resources explicitly
        # finally, substitute with statement with the creation of an object.
        with tempfile.TemporaryDirectory(prefix=self.experiment.name + '_',
                                         dir=self.tmp_dir) as workdirname:
            log.debug("## New temp consumer context: %s", workdirname)
            config_file = tempfile.NamedTemporaryFile(mode='w', prefix='trial_',
                                                      suffix='.conf', dir=workdirname,
                                                      delete=False)
            config_file.close()
            log.debug("## New temp config file: %s", config_file.name)
            results_file = tempfile.NamedTemporaryFile(mode='w', prefix='results_',
                                                       suffix='.out', dir=workdirname,
                                                       delete=False)
            results_file.close()
            log.debug("## New temp results file: %s", results_file.name)

            log.debug("## Building command line argument and configuration for trial.")
            cmd_args = self.template_builder.build_to(config_file.name,
                                                      self.current_trial,
                                                      self.experiment)

            command = [self.script_path] + cmd_args
            return self.interact_with_script(command, results_file)

    def interact_with_script(self, command, results_file):
        """Interact with user's script by launching it in a separate process.

        When the process exits, evaluation information
        reported to a file will be attempted to be retrieved.

        It sets ``self.current_trial.results``, if possible.

        Override it with a subclass of `Consumer` to implement a different
        way of communication with user's code and possibly management of the
        child process.

        :returns: Exit code of the child process
        :rtype: int

        """
        log.debug("## Launch user's script as a subprocess and wait for finish.")
        script_process = self.launch_process(command, results_file.name)

        if script_process.returncode is not None:
            return script_process.returncode

        try:
            returncode = script_process.wait()
        except KeyboardInterrupt:
            script_process.terminate()
            raise

        try:
            if returncode != 0:
                log.error("Something went wrong. Check logs. Process "
                          "returned with code %d !", returncode)
                if returncode == 2:
                    # This is the exit code returned when invalid options are given,
                    # for example when argparse fails
                    sys.exit(2)
        finally:
            log.debug("## Parse results from file and fill corresponding Trial object.")
            try:
                results = self.converter.parse(results_file.name)
                self.current_trial.results = [Trial.Result(name=res['name'],
                                                           type=res['type'],
                                                           value=res['value']) for res in results]
            except ValueError:  # JSON error because file is empty
                pass

        return returncode

    @staticmethod
    def launch_process(command, results_filename):
        """Facilitate launching a black-box trial.

        :returns: Child `subprocess.Popen` object

        """
        env = dict(os.environ)
        env['ORION_RESULTS_PATH'] = str(results_filename)
        process = subprocess.Popen(command, env=env)
        returncode = process.poll()
        if returncode is not None and returncode < 0:
            log.error("Failed to execute script to evaluate trial. Process "
                      "returned with code %d !", returncode)

        return process
