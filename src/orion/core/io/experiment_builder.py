# -*- coding: utf-8 -*-
# pylint:disable=protected-access,too-many-lines
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
import datetime
import getpass
import logging
import sys

from orion.algo.space import Space
import orion.core
from orion.core.evc.adapters import Adapter
from orion.core.evc.conflicts import detect_conflicts, ExperimentNameConflict
from orion.core.io import resolve_config
from orion.core.io.database import DuplicateKeyError
from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder
from orion.core.io.interactive_commands.branching_prompt import BranchingPrompt
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.exceptions import NoConfigurationError, RaceCondition
from orion.core.worker.experiment import Experiment, ExperimentView, ExperimentViewNew
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.core.worker.strategy import Strategy
from orion.storage.base import get_storage, Storage


log = logging.getLogger(__name__)


# pylint: disable=too-many-public-methods
class ExperimentBuilder(object):
    """To remove..."""

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

    # TODO: Remove
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

    # TODO: remove
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

    # TODO: Get rid of it
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
        """Create the storage instance from a configuration"""
        # TODO: move to utils.backward.
        #       This is required because orion.core.config can contain database at the root for
        #       backward compatibility.
        storage_config = config.get('protocol', {'type': 'legacy'})
        if 'database' in config:
            storage_config.setdefault('config', {})
            storage_config['config']['database'] = config['database']

        setup_storage(storage_config)


##
# Functions to build experiments
##

# TODO: branch_from cannot be passed from build_from_cmdargs, must add --branch-from argument
def build(name, version=None, branch_from=None, **config):
    """Build an experiment object

    If new, `space` argument must be provided, else all arguments are fetched from the database
    based on (name, version). If any argument given does not match the corresponding ones in the
    database for given (name, version), than the version is incremented and the experiment will be a
    child of the previous version.

    Parameters
    ----------
    name: str
        Name of the experiment to build
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.
    branch_from: str, optional
        Name of the experiment to branch from. The new experiment will have access to all trials
        from the parent experiment it has been branched from.
    space: dict, optional
        Optimization space of the algorithm. Should have the form `dict(name='<prior>(args)')`.
    algorithms: str or dict, optional
        Algorithm used for optimization.
    strategy: str or dict, optional
        Parallel strategy to use to parallelize the algorithm.
    max_trials: int, optional
        Maximum number or trials before the experiment is considered done.
    storage: dict, optional
        Configuration of the storage backend.

    """
    for key, value in config.items():
        if value is None:
            config.pop(key)

    if branch_from:
        new_name = name
        name = branch_from
    else:
        new_name = None

    db_config = fetch_config_from_db(name, version)

    config = resolve_config.merge_configs(db_config, config)

    config.setdefault('name', name)
    config.setdefault('version', version)

    if 'space' not in config:
        raise NoConfigurationError

    experiment = create_experiment(**config)
    if not experiment.id:
        experiment._init_done = True
        _register_experiment(experiment)
        return experiment

    conflicts = _get_conflicts(experiment)
    must_branch = len(conflicts.get()) > 1 or new_name
    if must_branch:
        branched_experiment = _branch_experiment(experiment, conflicts, new_name, version)
        _register_experiment(branched_experiment)
        return branched_experiment

    _update_experiment(experiment)
    return experiment


def build_view(name, version=None):
    """Build experiment view

    An experiment view provides all reading operations of standard experiment but prevents the
    modification of the experiment and its trials.

    Parameters
    ----------
    name: str
        Name of the experiment to build
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.

    """
    db_config = fetch_config_from_db(name, version)

    if not db_config:
        message = ("No experiment with given name '%s' and version '%s' inside database, "
                   "no view can be created." % (name, version if version else '*'))
        raise ValueError(message)

    if 'space' not in db_config and 'priors' in db_config.get('metadata', {}):
        db_config['space'] = db_config['metadata']['priors']

    # metadata = db_config['metadata']
    # if 'space' in db_config:
    #     space = db_config['space']
    # # TODO: Remove when space is ready
    # elif 'priors' in metadata:
    #     space = metadata['priors']
    # # Backward compatibility with v0.1.6
    # else:
    #     parser = OrionCmdlineParser(orion.core.config.user_script_config)
    #     parser.parse(metadata['user_args'])
    #     space = parser.priors

    experiment = create_experiment(**db_config)

    return ExperimentViewNew(experiment)


def create_experiment(name, version, space, **kwargs):
    """Instantiate the experiment and its attribute objects

    All unspecified arguments will be replaced by system's defaults (orion.core.config.*).

    Parameters
    ----------
    name: str
        Name of the experiment.
    version: int
        Version of the experiment.
    space: dict or Space object
        Optimization space of the algorithm. If dict, should have the form
        `dict(name='<prior>(args)')`.
    algorithms: str or dict, optional
        Algorithm used for optimization.
    strategy: str or dict, optional
        Parallel strategy to use to parallelize the algorithm.
    max_trials: int, optional
        Maximum number or trials before the experiment is considered done.
    storage: dict, optional
        Configuration of the storage backend.

    """
    experiment = Experiment(name=name, version=version)
    experiment._id = kwargs.get('_id', None)
    experiment.pool_size = kwargs.get('pool_size', orion.core.config.experiment.pool_size)
    experiment.max_trials = kwargs.get('max_trials', orion.core.config.experiment.max_trials)
    space = _instantiate_space(space)
    experiment.algorithms = _instantiate_algo(space, kwargs.get('algorithms'))
    experiment.producer = kwargs.get('producer', orion.core.config.experiment.producer.to_dict())
    experiment.producer['strategy'] = _instantiate_strategy(experiment.producer.get('stategy'))
    experiment.working_dir = kwargs.get('working_dir', orion.core.config.experiment.working_dir)
    experiment.metadata = kwargs.get('metadata', {'user': kwargs.get('user', getpass.getuser())})
    experiment.refers = kwargs.get('refers', {'parent_id': None, 'root_id': None, 'adapter': []})
    experiment.refers['adapter'] = _instantiate_adapters(experiment.refers.get('adapter', []))

    return experiment


def fetch_config_from_db(name, version=None):
    """Fetch configuration from database

    Parameters
    ----------
    name: str
        Name of the experiment to fetch
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.

    """
    configs = get_storage().fetch_experiments({'name': name})

    if not configs:
        return {}

    config = _fetch_config_version(configs, version)

    if len(configs) > 1:
        log.info("Many versions for experiment %s have been found. Using latest "
                 "version %s.", name, config['version'])

    if 'space' not in config and 'priors' in config.get('metadata', {}):
        config['space'] = config['metadata']['priors']

    return config


##
# Private helper functions to build experiments
##

def _instantiate_adapters(config):
    """Instantiate the adapter object

    Parameters
    ----------
    config: list
         List of adapter configurations to build a CompositeAdapter for the EVC.

    """
    return Adapter.build(config)


def _instantiate_space(config):
    """Instantiate the space object

    Build the Space object if argument is a dictionary, else return the Space object as is.

    Parameters
    ----------
    config: dict or Space object
        Dictionary of priors or already built Space object.

    """
    if isinstance(config, Space):
        return config

    return SpaceBuilder().build(config)


def _instantiate_algo(space, config):
    """Instantiate the algorithm object

    Parameters
    ----------
    config: dict, optional
        Configuration of the strategy. If None of empty, system's defaults are used
        (orion.core.config.experiment.algorithms).

    """
    if not config:
        config = orion.core.config.experiment.algorithms

    return PrimaryAlgo(space, config)


def _instantiate_strategy(config=None):
    """Instantiate the strategy object

    Parameters
    ----------
    config: dict, optional
        Configuration of the strategy. If None of empty, system's defaults are used
        (orion.core.config.producer.strategy).

    """
    if not config:
        config = orion.core.config.experiment.producer.strategy

    strategy_type, config = next(iter(config.items()))

    return Strategy(of_type=strategy_type, **config)


def _register_experiment(experiment):
    """Register a new experiment in the database"""
    experiment.metadata['datetime'] = datetime.datetime.utcnow()
    config = experiment.configuration
    # This will raise DuplicateKeyError if a concurrent experiment with
    # identical (name, metadata.user) is written first in the database.
    get_storage().create_experiment(config)

    # XXX: Reminder for future DB implementations:
    # MongoDB, updates an inserted dict with _id, so should you :P
    experiment._id = config['_id']

    # Update refers in db if experiment is root
    if experiment.refers.get('parent_id') is None:
        log.debug('update refers (name: %s)', experiment.name)
        experiment.refers['root_id'] = experiment._id
        get_storage().update_experiment(experiment, refers=experiment.configuration['refers'])


def _update_experiment(experiment):
    """Update experiment configuration in database"""
    log.debug('updating experiment (name: %s)', experiment.name)
    config = experiment.configuration

    # TODO: Remove since this should not occur anymore without metadata.user in the indices?
    # Writing the final config to an already existing experiment raises
    # a DuplicatKeyError because of the embedding id `metadata.user`.
    # To avoid this `final_config["name"]` is popped out before
    # `db.write()`, thus seamingly breaking  the compound index
    # `(name, metadata.user)`
    # config.pop("name")

    get_storage().update_experiment(experiment, **config)


def _branch_experiment(experiment, conflicts, branch_name, version):
    """Create a new branch experiment with adapters for the given conflicts"""
    experiment_brancher = ExperimentBranchBuilder(conflicts, branch_name)

    needs_manual_resolution = (not experiment_brancher.is_resolved or
                               experiment_brancher.manual_resolution)

    if not experiment_brancher.is_resolved:
        name_conflict = conflicts.get([ExperimentNameConflict])[0]
        if not name_conflict.is_resolved and not version:
            raise RaceCondition('There was likely a race condition during version increment.')

    if needs_manual_resolution:
        # TODO: This should only be possible when using cmdline API
        branching_prompt = BranchingPrompt(experiment_brancher)

        if not sys.__stdin__.isatty():
            raise ValueError(
                "Configuration is different and generates a branching event:\n{}".format(
                    branching_prompt.get_status()))

        branching_prompt.cmdloop()

        if branching_prompt.abort or not experiment_brancher.is_resolved:
            sys.exit()

    config = experiment_brancher.conflicting_config
    config['refers']['adapter'] = experiment_brancher.create_adapters()
    config['refers']['parent_id'] = experiment.id

    return create_experiment(**config)


def _get_conflicts(experiment):
    """Get conflicts between current experiment and corresponding configuration in database"""
    db_experiment = build_view(experiment.name, experiment.version)
    # TODO: Remove when _init_done is removed.
    experiment._init_done = True
    db_experiment._experiment._init_done = True
    conflicts = detect_conflicts(db_experiment.configuration, experiment.configuration)

    # elif must_branch and not enable_branching:
    #     raise ValueError("Configuration is different and generate a branching event")

    return conflicts


def _fetch_config_version(configs, version=None):
    """Fetch the experiment configuration corresponding to the given version

    Parameters
    ----------
    configs: list
        List of configurations fetched from storoge.
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.

    """
    max_version = max(configs, key=lambda exp: exp.get('version', 1)).get('version', 1)

    if version is None:
        version = max_version
    else:
        version = version

    if version > max_version:
        log.warning("Version %s was specified but most recent version is only %s. "
                    "Using %s.", version, max_version, max_version)

    version = min(version, max_version)

    configs = filter(lambda exp: exp.get('version', 1) == version, configs)

    return next(iter(configs))


def setup_storage(storage=None):
    """Create the storage instance from a configuration.

    Parameters
    ----------
    config: dict
        Configuration for the storage backend.

    """
    if storage is None:
        storage = {'type': 'legacy'}
        # TODO: storage = orion.core.config.storage.to_dict()

    if storage['type'] == 'legacy':
        storage.setdefault('config', {})

        if 'database' not in storage['config']:
            storage['config']['database'] = orion.core.config.storage.database.to_dict()

    storage_type = storage.pop('type')

    log.debug("Creating %s storage client with args: %s", storage_type, storage)
    try:
        Storage(of_type=storage_type, **storage)
    except ValueError:
        if Storage().__class__.__name__.lower() != storage_type.lower():
            raise


###
# Functions for commandline API
###

def build_from_args(cmdargs):
    """Build an experiment based on commandline arguments.

    Options provided in commandline and configuration file (--config) will overwrite system's
    default values and configuration from database if experiment already exits.
    Commandline arguments have precedence over configuration file options.

    .. seealso::

        :func:`orion.core.io.experiment_builder.build` for more information on experiment creation.

    """
    cmd_config = get_cmd_config(cmdargs)

    storage = cmd_config.get('storage')

    setup_storage(storage)

    if 'priors' in cmd_config['metadata']:
        cmd_config['space'] = cmd_config['metadata']['priors']

    return build(**cmd_config)


def build_view_from_args(cmdargs):
    """Build an experiment view based on commandline arguments

    .. seealso::

        :func:`orion.core.io.experiment_builder.build_view` for more information on experiment view
        creation.

    """
    cmd_config = get_cmd_config(cmdargs)

    storage = cmd_config.get('storage')

    setup_storage(storage)

    name = cmd_config.get('name')
    version = cmd_config.get('version')

    return build_view(name, version)


def get_cmd_config(cmdargs):
    """Fetch configuration defined by commandline and local configuration file.

    Arguments of commandline have priority over options in configuration file.

    """
    cmd_config = resolve_config.fetch_config(cmdargs)

    metadata = dict(metadata=resolve_config.fetch_metadata(cmdargs))

    cmd_config = resolve_config.merge_configs(cmd_config, cmdargs, metadata)

    if 'user' in cmd_config:
        cmd_config['metadata']['user'] = cmd_config['user']

    # TODO: Find a better solution
    # if isinstance(cmd_config['algorithms'], dict) and len(cmd_config['algorithms']) > 1:
    #     for key in list(config_from_db['algorithms'].keys()):
    #         cmd_config['algorithms'].pop(key)

    if 'user_args' in cmd_config['metadata']:
        parser = OrionCmdlineParser(orion.core.config.user_script_config)
        parser.parse(cmd_config['metadata']['user_args'])
        cmd_config['metadata'].update({
            'parser': parser.get_state_dict(),
            'priors': dict(parser.priors)})

    return cmd_config
