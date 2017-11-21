# -*- coding: utf-8 -*-
"""
mopt:
  MetaOpt cli script for asynchronous distributed optimization

ERROR CODES
-----------
* 0  : Success
* -1 : Provide a name for the experiment (cli argument or moptconfig file)

"""
from __future__ import absolute_import

import datetime
import getpass
import logging
import sys

from metaopt import resolve_config


def main():
    """Entry point for metaopt.core functionality."""
    starttime = datetime.datetime.utcnow()
    user = getpass.getuser()

    expconfig, moptdb = infer_config_and_db(user)
    expmetadata = infer_metadata(user, starttime)
    # XXX: More metadata on this experiment, mby for each run of this experiment
    # log different configs too..

    # XXX Module defaults should be written to an example configuration file
    # automatically as pre-commit hooks!!
    # 1. check whether supplied method{optimizer, dynamic} names exist
    # 2. check whether supplied specific methods/parameters are correct
    print(moptdb)
    print()
    print(dict(expconfig))
    print()
    print(expmetadata)


def infer_config_and_db(user):
    """Use metaopt.resolve_config to organize how configuration is built."""
    # Fetch experiment name, user's script, args and parameter config
    # Use `-h` option to show help
    cmdargs, cmdconfig = resolve_config.fetch_mopt_args(__doc__)
    #  print(cmdargs, cmdconfig)

    expconfig = resolve_config.fetch_default_options()
    # Fetch mopt system variables (database and resource information)
    # See :const:`metaopt.io.resolve_config.ENV_VARS` for environmental variables used
    expconfig = resolve_config.merge_env_vars(expconfig)

    tmpconfig = resolve_config.merge_mopt_config(expconfig, dict(),
                                                 cmdconfig, cmdargs)
    db_opts = tmpconfig['database']
    # (TODO) Init database with `db_opts`
    print(user)
    moptdb = object()

    exp_name = tmpconfig['exp_name']
    # (TODO) Get experiment metadata for experiment with name == `exp_name`,
    # if it exists.
    dbconfig = dict()

    expconfig = resolve_config.merge_mopt_config(expconfig, dbconfig,
                                                 cmdconfig, cmdargs)
    exp_name = expconfig['exp_name']

    print(db_opts)
    print()
    print(exp_name)
    if exp_name is None:
        logging.fatal("Could not infer experiment's name:\n"
                      "Please use cmd arg or a cmd provided config file.")
        sys.exit(-1)

    return expconfig, moptdb


def infer_metadata(user, starttime):
    """Identify current experiment in terms of user, time and code base."""
    metadata = dict()
    metadata['user'] = user
    metadata['starttime'] = starttime
    return metadata
