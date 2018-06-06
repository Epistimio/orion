#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.init_only` -- Module running the init_only command
=======================================================================

.. module:: init_only
   :platform: Unix
   :synopsis: Creates a new experiment.
"""

import logging

import orion
from orion.core.io import resolve_config
from orion.core.cli import base as cli
from orion.core.io.database import Database, DuplicateKeyError
from orion.core.worker.experiment import Experiment

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    init_only_parser = parser.add_parser('init_only', help='init_only help')

    cli.get_basic_args_group(init_only_parser)

    cli.get_user_args_group(init_only_parser)

    init_only_parser.set_defaults(func=main)

    return init_only_parser


def main(args):
    """Fetch config and initialize experiment"""
    # Note: Side effects on args
    config = fetch_config(args)

    _execute(args, config)


def fetch_config(args):
    """Create the dictionary of modified args for the execution of the command"""
    # Explicitly add orion's version as experiment's metadata
    args['metadata'] = dict()
    args['metadata']['orion_version'] = orion.core.__version__
    log.debug("Using orion version %s", args['metadata']['orion_version'])

    config = resolve_config.fetch_config(args)

    args.pop('user_script')
    args['metadata']['user_args'] = args.pop('user_args')

    return config


# By inferring the experiment, we create a new configured experiment
def _execute(cmdargs, cmdconfig):
    _infer_experiment(cmdargs, cmdconfig)


def _infer_experiment(cmdargs, cmdconfig):
    # Initialize configuration dictionary.
    # Fetch info from defaults and configurations from default locations.
    expconfig = resolve_config.fetch_default_options()

    # Fetch orion system variables (database and resource information)
    # See :const:`orion.core.io.resolve_config.ENV_VARS` for environmental
    # variables used
    expconfig = resolve_config.merge_env_vars(expconfig)

    # Initialize singleton database object
    tmpconfig = resolve_config.merge_orion_config(expconfig, dict(),
                                                  cmdconfig, cmdargs)
    db_opts = tmpconfig['database']
    dbtype = db_opts.pop('type')

    log.debug("Creating %s database client with args: %s", dbtype, db_opts)
    Database(of_type=dbtype, **db_opts)

    # Information should be enough to infer experiment's name.
    exp_name = tmpconfig['name']
    if exp_name is None:
        raise RuntimeError("Could not infer experiment's name. "
                           "Please use either `name` cmd line arg or provide "
                           "one in orion's configuration file.")

    experiment = create_experiment(exp_name, expconfig, cmdconfig, cmdargs)

    return experiment


def create_experiment(exp_name, expconfig, cmdconfig, cmdargs):
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

    expconfig = resolve_config.merge_orion_config(expconfig, dbconfig,
                                                  cmdconfig, cmdargs)

    # Infer rest information about the process + versioning
    expconfig['metadata'] = infer_versioning_metadata(expconfig['metadata'])

    # Pop out configuration concerning databases and resources
    expconfig.pop('database', None)
    expconfig.pop('resources', None)

    # Finish experiment's configuration and write it to database.
    try:
        experiment.configure(expconfig)
    except DuplicateKeyError:
        # Fails if concurrent experiment with identical (name, metadata.user)
        # is written first in the database.
        # Next infer_experiment() should either load experiment from database
        # and run smoothly if identical or trigger an experiment fork.
        # In other words, there should not be more than 1 level of recursion.
        experiment = create_experiment(exp_name, expconfig, cmdconfig, cmdargs)

    return experiment


def infer_versioning_metadata(existing_metadata):
    """Infer information about user's script versioning if available."""
    # VCS system
    # User repo's version
    # User repo's HEAD commit hash
    return existing_metadata
