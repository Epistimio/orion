# -*- coding: utf-8 -*-
# pylint: disable=eval-used,protected-access
"""
:mod:`orion.core.cli.insert` -- Module to insert new trials
===========================================================

.. module:: insert
   :platform: Unix
   :synopsis: Insert creates new trials for a given experiment with fixed values

"""
import logging

import orion
from orion.core.cli import resolve_config
from orion.core.io.database import Database
from orion.core.utils import get_qualified_name
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.viz.analysers import AnalyserWrapper
from orion.viz.plotters import PlotterWrapper

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    insert_parser = parser.add_parser('viz', help='viz help')

    resolve_config.get_basic_args_group(insert_parser)

    insert_parser.add_argument('--analyser', type=str)
    insert_parser.add_argument('--plotter', type=str)

    insert_parser.set_defaults(func=main)

    return insert_parser


def main(args):
    """Fetch config and insert new point"""
    # Note: Side effects on args
    config = fetch_config(args)

    _execute(args, config)


def fetch_config(args):
    """Get options from command line arguments."""
    # Explicitly add orion's version as experiment's metadata
    args['metadata'] = dict()
    args['metadata']['orion_version'] = orion.core.__version__
    log.debug("Using orion version %s", args['metadata']['orion_version'])

    config = resolve_config.fetch_config(args)

    return config


def _execute(cmd_args, file_config):
    analyser_config = file_config.pop('analyser')
    plotter_config = file_config.pop('plotter')
    experiment = _infer_experiment(cmd_args, file_config)

    if experiment.id is None:
        raise ValueError("No experiment with given name '%s' for user '%s' inside database, "
                         "can't insert." % (experiment.name, experiment.metadata['user']))

    trials = Trial.build(experiment._db.read('trials', dict(experiment=experiment.id)))
    analyser = AnalyserWrapper(trials, experiment, analyser_config)

    print(analyser_config)
    plotter = PlotterWrapper(analyser.analyse(), ['png'], plotter_config)
    plotter.plot()


def _infer_experiment(cmd_args, file_config):
    # Initialize configuration dictionary.
    # Fetch info from defaults and configurations from default locations.
    expconfig = resolve_config.fetch_default_options()

    # Fetch orion system variables (database and resource information)
    # See :const:`orion.core.io.resolve_config.ENV_VARS` for environmental
    # variables used
    expconfig = resolve_config.merge_env_vars(expconfig)

    # Initialize singleton database object
    tmpconfig = resolve_config.merge_orion_config(expconfig, dict(),
                                                  file_config, cmd_args)

    db_opts = tmpconfig['database']
    dbtype = db_opts.pop('type')

    log.debug("Creating %s database client with args: %s", dbtype, db_opts)
    Database(of_type=(get_qualified_name('orion.core.io.database', dbtype), dbtype), **db_opts)

    # Information should be enough to infer experiment's name.
    exp_name = tmpconfig['name']
    if exp_name is None:
        raise RuntimeError("Could not infer experiment's name. "
                           "Please use either `name` cmd line arg or provide "
                           "one in orion's configuration file.")

    experiment = create_experiment(exp_name, expconfig, file_config, cmd_args)

    return experiment


def create_experiment(exp_name, expconfig, file_config, cmd_args):
    """Create an experiment based on configuration.

    Configuration is a combination of command line, experiment configuration
    file, experiment configuration in database and orion configuration files.

    Precedence of configurations is:
    `cmdargs` > `cmdconfig` > `dbconfig` > `expconfig`

    This means `expconfig` values would be overwritten by `dbconfig` and so on.

    Parameters
    ----------
    exp_name: str
        Name of the experiment
    expconfig: dict
        Configuration coming from default configuration files.
    cmdconfig: dict
        Configuration coming from configuration file.
    cmdargs: dict
        Configuration coming from command line arguments.

    """
    # Initialize experiment object.
    # Check for existing name and fetch configuration.
    experiment = Experiment(exp_name)
    dbconfig = experiment.configuration

    log.debug("DB config")
    log.debug(dbconfig)

    expconfig = resolve_config.merge_orion_config(expconfig, dbconfig,
                                                  file_config, cmd_args)
    # Infer rest information about the process + versioning
    expconfig['metadata'] = infer_versioning_metadata(expconfig['metadata'])

    # Pop out configuration concerning databases and resources
    expconfig.pop('database', None)
    expconfig.pop('resources', None)

    log.info(expconfig)

    experiment.configure(expconfig)

    return experiment


def infer_versioning_metadata(existing_metadata):
    """Infer information about user's script versioning if available."""
    # VCS system
    # User repo's version
    # User repo's HEAD commit hash
    return existing_metadata
