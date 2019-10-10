# -*- coding: utf-8 -*-
# pylint:disable=protected-access
"""
:mod:`orion.core.io.experiment_builder` -- Create experiment from user options
==============================================================================

.. module:: experiment
   :platform: Unix
   :synopsis: Functions which build `Experiment` and `ExperimentView` objects
       based on user configuration.


The instantiation of an `Experiment` is not a trivial process when the user request an experiment
with specific options. One can easily create a new experiment with
`ExperimentView('some_experiment_name')`, but the configuration of a _writable_ experiment is less
straighforward. This is because there is many sources of configuration and they have a strict
hierarchy. From the more global to the more specific, there is:

1. Global configuration:

  Defined by `orion.core.DEF_CONFIG_FILES_PATHS`.
  Can be scattered in user file system, defaults could look like:

    - `/some/path/to/.virtualenvs/orion/share/orion.core`
    - `/etc/xdg/xdg-ubuntu/orion.core`
    - `/home/${USER}/.config/orion.core`

  Note that some variables have default value even if user do not defined them in global
  configuration:

    - `max_trials = orion.core.io.resolve_config.DEF_CMD_MAX_TRIALS`
    - `pool_size = orion.core.io.resolve_config.DEF_CMD_POOL_SIZE`
    - `algorithms = random`
    - Database specific:

      * `database.name = 'orion'`
      * `database.type = 'MongoDB'`
      * `database.host = ${HOST}`

2. Oríon specific environment variables:

   Environment variables which can override global configuration

    - Database specific:

      * `ORION_DB_NAME`
      * `ORION_DB_TYPE`
      * `ORION_DB_ADDRESS`

3. Experiment configuration inside the database

  Configuration of the experiment if present in the database.
  Making this part of the configuration of the experiment makes it possible
  for the user to execute an experiment by only specifying partial configuration. The rest of the
  configuration is fetched from the database.

  For example, a user could:

    1. Rerun the same experiment

      Only providing the name is sufficient to rebuild the entire configuration of the
      experiment.

    2. Make a modification to an existing experiment

      The user can provide the name of the existing experiment and only provide the changes to
      apply on it. Here is an minimal example where we fully initialize a first experiment with a
      config file and then branch from it with minimal information.

      .. code-block:: bash

          # Initialize root experiment
          orion init_only --config previous_exeriment.yaml ./userscript -x~'uniform(0, 10)'
          # Branch a new experiment
          orion hunt -n previous_experiment ./userscript -x~'uniform(0, 100)'

4. Configuration file

  This configuration file is meant to overwrite the configuration coming from the database.
  If this configuration file was interpreted as part of the global configuration, a user could
  only modify an experiment using command line arguments.

5. Command-line arguments

  Those are the arguments provided to `orion` for any method (hunt, insert, etc). It includes the
  argument to `orion` itself as well as the user's script name and its arguments.

"""
import copy
import logging

import orion.core
from orion.core.io import resolve_config
from orion.core.io.database import DuplicateKeyError
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser
from orion.core.utils.exceptions import NoConfigurationError, RaceCondition
from orion.core.worker.experiment import Experiment, ExperimentView
from orion.storage.base import Storage


log = logging.getLogger(__name__)


# pylint: disable=too-many-public-methods
class ExperimentBuilder(object):
    """Builder for :class:`orion.core.worker.experiment.Experiment`
    and :class:`orion.core.worker.experiment.ExperimentView`

    .. seealso::

        `orion.core.io.experiment_builder` for more information on the process of building
        experiments.

        :class:`orion.core.worker.experiment.Experiment`
        :class:`orion.core.worker.experiment.ExperimentView`
    """

    # pylint:disable=no-self-use
    def fetch_default_options(self):
        """Get dictionary of default options"""
        return resolve_config.fetch_default_options()

    # pylint:disable=no-self-use
    def fetch_env_vars(self):
        """Get dictionary of environment variables specific to Oríon"""
        return resolve_config.fetch_env_vars()

    def fetch_file_config(self, cmdargs):
        """Get dictionary of options from configuration file provided in command-line"""
        return resolve_config.fetch_config(cmdargs)

    def fetch_config_from_db(self, cmdargs):
        """Get dictionary of options from experiment found in the database

        Notes
        -----
            This method builds an experiment view in the background to fetch the configuration from
            the database.

        """
        try:
            experiment_view = self.build_view_from(cmdargs)
        except ValueError as e:
            if "No experiment with given name" in str(e):
                return {}
            raise

        return experiment_view.configuration

    def fetch_metadata(self, cmdargs):
        """Infer rest information about the process + versioning"""
        return resolve_config.fetch_metadata(cmdargs)

    def fetch_full_config(self, cmdargs, use_db=True):
        """Get dictionary of the full configuration of the experiment.

        .. seealso::

            `orion.core.io.experiment_builder` for more information on the hierarchy of
            configurations.

        Parameters
        ----------
        cmdargs:

        use_db: bool
            Use experiment configuration found in database if True. Defaults to True.

        Notes
        -----
            This method builds an experiment view in the background to fetch the configuration from
            the database.

        """
        default_options = self.fetch_default_options()
        env_vars = self.fetch_env_vars()
        if use_db:
            config_from_db = self.fetch_config_from_db(cmdargs)
        else:
            config_from_db = {}
        cmdconfig = self.fetch_file_config(cmdargs)
        metadata = dict(metadata=self.fetch_metadata(cmdargs))

        exp_config = resolve_config.merge_configs(
            default_options, env_vars, copy.deepcopy(config_from_db), cmdconfig, cmdargs, metadata)

        if 'user' in exp_config:
            exp_config['metadata']['user'] = exp_config['user']

        # TODO: Find a better solution
        if isinstance(exp_config['algorithms'], dict) and len(exp_config['algorithms']) > 1:
            for key in list(config_from_db['algorithms'].keys()):
                exp_config['algorithms'].pop(key)

        return exp_config

    def build_view_from(self, cmdargs):
        """Build an experiment view based on full configuration.

        .. seealso::

            `orion.core.io.experiment_builder` for more information on the hierarchy of
            configurations.

            :class:`orion.core.worker.experiment.ExperimentView` for more information on the
            experiment view object.
        """
        local_config = self.fetch_full_config(cmdargs, use_db=False)

        self.setup_storage(local_config)

        # Information should be enough to infer experiment's name.
        exp_name = local_config['name']
        if exp_name is None:
            raise RuntimeError("Could not infer experiment's name. "
                               "Please use either `name` cmd line arg or provide "
                               "one in orion's configuration file.")

        name = local_config['name']
        user = local_config.get('user', None)
        version = local_config.get('version', None)
        return ExperimentView(name, user=user, version=version)

    def build_from(self, cmdargs, handle_racecondition=True):
        """Build a fully configured (and writable) experiment based on full configuration.

        .. seealso::

            `orion.core.io.experiment_builder` for more information on the hierarchy of
            configurations.

            :class:`orion.core.worker.experiment.Experiment` for more information on the experiment
            object.
        """
        full_config = self.fetch_full_config(cmdargs)

        log.info(full_config)

        try:
            experiment = self.build_from_config(full_config)
        except (DuplicateKeyError, RaceCondition):
            # Fails if concurrent experiment with identical (name, version)
            # is written first in the database.
            # Next build_from(cmdargs) should either load experiment from database
            # and run smoothly if identical or trigger an experiment fork.
            # In other words, there should not be more than 1 level of recursion.
            if handle_racecondition:
                experiment = self.build_from(cmdargs, handle_racecondition=False)

            raise

        return experiment

    def build_from_config(self, config):
        """Build a fully configured (and writable) experiment based on full configuration.

        .. seealso::

            `orion.core.io.experiment_builder` for more information on the hierarchy of
            configurations.

            :class:`orion.core.worker.experiment.Experiment` for more information on the experiment
            object.
        """
        log.info(config)

        # Pop out configuration concerning databases and resources
        config.pop('database', None)
        config.pop('resources', None)

        experiment = Experiment(config['name'], config.get('user', None),
                                config.get('version', None))

        # TODO: Handle both from cmdline and python APIs.
        if 'priors' not in config['metadata'] and 'user_args' not in config['metadata']:
            raise NoConfigurationError

        # Parse to generate priors
        if 'user_args' in config['metadata']:
            parser = OrionCmdlineParser(orion.core.config.user_script_config)
            parser.parse(config['metadata']['user_args'])
            config['metadata']['parser'] = parser.get_state_dict()
            config['metadata']['priors'] = dict(parser.priors)

        # Finish experiment's configuration and write it to database.
        experiment.configure(config)

        return experiment

    def setup_storage(self, config):
        """Create the storage instance from a configuration.

        Parameters
        ----------
        config: dict
            Configuration for the database.

        """
        # TODO: Fix this in config refactoring
        storage_opts = config.get('protocol', {'type': 'legacy'})
        storage_type = storage_opts.pop('type')

        log.debug("Creating %s storage client with args: %s", storage_type, storage_opts)
        try:
            Storage(of_type=storage_type, config=config, **storage_opts)
        except ValueError:
            if Storage().__class__.__name__.lower() != storage_type.lower():
                raise
