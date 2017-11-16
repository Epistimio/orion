# -*- coding: utf-8 -*-
"""
mopt:
  MetaOpt cli script for asynchronous distributed optimization

"""
from __future__ import absolute_import

import datetime
import getpass

from metaopt import resolve_config as resolvconf


def main():
    """Entry point for metaopt.core functionality."""
    starttime = datetime.datetime.utcnow()
    user = getpass.getuser()

    expconfig, moptdb = infer_config_and_db(user, starttime)
    expmetadata = infer_metadata(user, starttime)
    # XXX: More metadata on this experiment, mby for each run of this experiment
    # log different configs too..

    # XXX Module defaults should be written to an example configuration file
    # automatically as pre-commit hooks!!
    # 1. check whether supplied method{optimizer, dynamic} names exist
    # 2. check whether supplied specific methods/parameters are correct
    print(dict(expconfig))
    print()
    print(expmetadata)


def infer_config_and_db(user, starttime):
    """Use metaopt.resolve_config to organize how configuration is built."""
    # Fetch experiment name, user's script, args and parameter config
    # Use `-h` option to show help
    cmdargs, cmdconfig = resolvconf.mopt_args(__doc__)
    #  print(cmdargs, cmdconfig)

    expconfig = resolvconf.default_options(user, starttime)
    # Fetch mopt system variables (database and resource information)
    # See :const:`metaopt.io.resolvconf.ENV_VARS` for environmental variables used
    expconfig = resolvconf.env_vars(expconfig)

    # (TODO) Init database with `db_opts`
    tmpconfig = resolvconf.mopt_config(expconfig, dict(), cmdconfig, cmdargs)
    db_opts = tmpconfig['database']
    moptdb = object()

    # (TODO) Get experiment metadata for experiment with name == `exp_name`,
    # if it exists.
    exp_name = tmpconfig['exp_name']
    dbconfig = dict()

    print(db_opts)
    print()
    print(exp_name)

    expconfig = resolvconf.mopt_config(expconfig, dbconfig, cmdconfig, cmdargs)

    return expconfig, moptdb


def infer_metadata(user, starttime):
    """Identify current experiment in terms of user, time and code base."""
    metadata = dict()
    metadata['user'] = user
    metadata['starttime'] = starttime
    return metadata
