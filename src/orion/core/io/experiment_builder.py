# -*- coding: utf-8 -*-
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
from orion.core.io.space_builder import SpaceBuilder
import orion.core.utils.backward as backward
from orion.core.utils.exceptions import (
    BranchingEvent, NoConfigurationError, NoNameError, RaceCondition)
from orion.core.worker.experiment import Experiment, ExperimentView
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.core.worker.strategy import Strategy
from orion.storage.base import get_storage, setup_storage


log = logging.getLogger(__name__)


##
# Functions to build experiments
##

def build(name, version=None, branching=None, **config):
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
    branching: dict, optional
        Arguments to control the branching.

        branch_from: str, optional
            Name of the experiment to branch from.
        manual_resolution: bool, optional
            Starts the prompt to resolve manually the conflicts. Defaults to False.
        non_monitored_arguments: list of str, optional
            Will ignore these arguments while looking for differences. Defaults to [].
        ignore_code_changes: bool, optional
            Will ignore code changes while looking for differences. Defaults to False.
        algorithm_change: bool, optional
            Whether to automatically solve the algorithm conflict (change of algo config).
            Defaults to True.
        code_change_type: str, optional
            How to resolve code change automatically. Must be one of 'noeffect', 'unsure' or
            'break'.  Defaults to 'break'.
        cli_change_type: str, optional
            How to resolve cli change automatically. Must be one of 'noeffect', 'unsure' or 'break'.
            Defaults to 'break'.
        config_change_type: str, optional
            How to resolve config change automatically. Must be one of 'noeffect', 'unsure' or
            'break'.  Defaults to 'break'.

    """
    config = copy.deepcopy(config)
    for key, value in list(config.items()):
        if key.startswith('_') or value is None:
            config.pop(key)

    if 'strategy' in config:
        config['producer'] = {'strategy': config.pop('strategy')}

    if branching is None:
        branching = {}

    if branching.get('branch_from'):
        branching.setdefault('branch_to', name)
        name = branching['branch_from']

    db_config = fetch_config_from_db(name, version)

    new_config = config
    config = resolve_config.merge_configs(db_config, config)

    metadata = resolve_config.fetch_metadata(config.get('user'), config.get('user_args'))

    config = resolve_config.merge_configs(db_config, config, {'metadata': metadata})

    # TODO: Find a better solution
    if isinstance(config.get('algorithms'), dict) and len(config['algorithms']) > 1:
        config['algorithms'] = new_config['algorithms']

    config.setdefault('name', name)
    config.setdefault('version', version)

    if 'space' not in config:
        raise NoConfigurationError(
            'Experiment {} does not exist in DB and space was not defined.'.format(name))

    if len(config['space']) == 0:
        raise NoConfigurationError("No prior found. Please include at least one.")

    experiment = create_experiment(**copy.deepcopy(config))
    if experiment.id is None:
        try:
            _register_experiment(experiment)
        except DuplicateKeyError:
            experiment = build(branching=branching, **config)

        return experiment

    conflicts = _get_conflicts(experiment, branching)
    must_branch = len(conflicts.get()) > 1 or branching.get('branch_to')
    if must_branch:
        branched_experiment = _branch_experiment(experiment, conflicts, version, branching)
        try:
            _register_experiment(branched_experiment)
        except DuplicateKeyError as e:
            raise RaceCondition('There was a race condition during branching.') from e

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

    db_config.setdefault('version', 1)

    experiment = create_experiment(**db_config)

    return ExperimentView(experiment)


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
    experiment._id = kwargs.get('_id', None)  # pylint:disable=protected-access
    experiment.pool_size = kwargs.get('pool_size')
    if experiment.pool_size is None:
        experiment.pool_size = orion.core.config.experiment.get(
            'pool_size', deprecated='ignore')
    experiment.max_trials = kwargs.get('max_trials', orion.core.config.experiment.max_trials)
    experiment.space = _instantiate_space(space)
    experiment.algorithms = _instantiate_algo(experiment.space, kwargs.get('algorithms'))
    experiment.producer = kwargs.get('producer', {})
    experiment.producer['strategy'] = _instantiate_strategy(experiment.producer.get('strategy'))
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

    backward.populate_space(config)

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
        config = orion.core.config.experiment.strategy

    if isinstance(config, str):
        strategy_type = config
        config = {}
    else:
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
    experiment._id = config['_id']  # pylint:disable=protected-access

    # Update refers in db if experiment is root
    if experiment.refers.get('parent_id') is None:
        log.debug('update refers (name: %s)', experiment.name)
        experiment.refers['root_id'] = experiment.id
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
    config.pop("name")

    get_storage().update_experiment(experiment, **config)


def _branch_experiment(experiment, conflicts, version, branching_arguments):
    """Create a new branch experiment with adapters for the given conflicts"""
    experiment_brancher = ExperimentBranchBuilder(conflicts, **branching_arguments)

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
            raise BranchingEvent(branching_prompt.get_status())

        branching_prompt.cmdloop()

        if branching_prompt.abort or not experiment_brancher.is_resolved:
            sys.exit()

    config = experiment_brancher.conflicting_config
    config['refers']['adapter'] = experiment_brancher.create_adapters().configuration
    config['refers']['parent_id'] = experiment.id

    config.pop('_id')

    return create_experiment(**config)


def _get_conflicts(experiment, branching):
    """Get conflicts between current experiment and corresponding configuration in database"""
    db_experiment = build_view(experiment.name, experiment.version)
    conflicts = detect_conflicts(db_experiment.configuration, experiment.configuration,
                                 branching)

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

    if 'name' not in cmd_config:
        raise NoNameError()

    setup_storage(cmd_config['storage'], debug=cmd_config.get('debug'))

    return build(**cmd_config)


def build_view_from_args(cmdargs):
    """Build an experiment view based on commandline arguments

    .. seealso::

        :func:`orion.core.io.experiment_builder.build_view` for more information on experiment view
        creation.

    """
    cmd_config = get_cmd_config(cmdargs)

    if 'name' not in cmd_config:
        raise NoNameError()

    setup_storage(cmd_config['storage'], debug=cmd_config.get('debug'))

    name = cmd_config.get('name')
    version = cmd_config.get('version')

    return build_view(name, version)


def get_cmd_config(cmdargs):
    """Fetch configuration defined by commandline and local configuration file.

    Arguments of commandline have priority over options in configuration file.
    """
    cmdargs = resolve_config.fetch_config_from_cmdargs(cmdargs)
    cmd_config = resolve_config.fetch_config(cmdargs)
    cmd_config = resolve_config.merge_configs(cmd_config, cmdargs)

    cmd_config.update(cmd_config.pop('experiment', {}))
    cmd_config['branching'] = cmd_config.pop('evc', {})

    metadata = resolve_config.fetch_metadata(cmd_config.get('user'), cmd_config.get('user_args'))
    cmd_config['metadata'] = metadata
    cmd_config.pop('config', None)

    backward.populate_space(cmd_config)
    backward.update_db_config(cmd_config)

    return cmd_config
