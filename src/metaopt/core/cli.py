#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`metaopt.core.cli` -- Functions that define console scripts
================================================================

.. module:: cli
   :platform: Unix
   :synopsis: Helper functions to setup an experiment and execute it.

"""
import logging

from metaopt.core import resolve_config
from metaopt.core.io.database import Database
from metaopt.core.worker import workon
from metaopt.core.worker.experiment import Experiment

log = logging.getLogger(__name__)

CLI_DOC_HEADER = """
mopt:
  MetaOpt cli script for asynchronous distributed optimization

"""


def main():
    """Entry point for `metaopt.core` functionality."""
    experiment = infer_experiment()
    workon(experiment)
    return 0


def infer_experiment():
    """Use `metaopt.core.resolve_config` to organize how configuration is built."""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help
    cmdargs, cmdconfig = resolve_config.fetch_mopt_args(CLI_DOC_HEADER)

    # Initialize configuration dictionary.
    # Fetch info from defaults and configurations from default locations.
    expconfig = resolve_config.fetch_default_options()

    # Fetch mopt system variables (database and resource information)
    # See :const:`metaopt.core.io.resolve_config.ENV_VARS` for environmental variables used
    expconfig = resolve_config.merge_env_vars(expconfig)

    # Initialize singleton database object
    tmpconfig = resolve_config.merge_mopt_config(expconfig, dict(),
                                                 cmdconfig, cmdargs)
    db_opts = tmpconfig['database']
    dbtype = db_opts.pop('type')
    log.debug("Creating %s database client with args: %s", dbtype, db_opts)
    Database(of_type=dbtype, **db_opts)

    # Information should be enough to infer experiment's name.
    exp_name = tmpconfig['name']
    if exp_name is None:
        raise RuntimeError("Could not infer experiment's name. "
                           "Please use either `name` cmd line arg or provide one "
                           "in metaopt's configuration file.")

    # Initialize experiment object.
    # Check for existing name and fetch configuration.
    experiment = Experiment(exp_name)
    dbconfig = experiment.configuration

    expconfig = resolve_config.merge_mopt_config(expconfig, dbconfig,
                                                 cmdconfig, cmdargs)

    # Infer rest information about the process + versioning
    expconfig['metadata'] = infer_versioning_metadata(expconfig['metadata'])

    # Pop out configuration concerning databases and resources
    expconfig.pop('database', None)
    expconfig.pop('resources', None)
    expconfig.pop('status', None)

    # Finish experiment's configuration
    experiment.configure(expconfig)

    return experiment


def infer_versioning_metadata(existing_metadata):
    """Infer information about user's script versioning if available."""
    # VCS system
    # User repo's version
    # User repo's HEAD commit hash
    return existing_metadata


if __name__ == "__main__":
    main()
