# -*- coding: utf-8 -*-
"""
:mod:`metaopt.resolve_config` -- Configuration parsing and resolving
====================================================================

.. module:: resolve_config
   :platform: Unix
   :synopsis: How does metaopt resolve configuration settings?

How:

 - Experiment name resolves like this:
    * cmd-arg **>** cmd-provided moptconfig **>** REQUIRED (no default is given)

 - Database options resolve with the following precedence (high to low):
    * cmd-provided moptconfig **>** env vars **>** default files **>** defaults

.. seealso:: :const:`ENV_VARS`, :const:`ENV_VARS_DB`


 - All other managerial, `Optimization` or `Dynamic` options resolve like this:

    * cmd-args **>** cmd-provided moptconfig **>** database (if experiment name
      can be found) **>** default files

Default files are given as a list at :const:`DEF_CONFIG_FILES_PATHS` and a
precedence is respected when building the settings dictionary:

 * default metaopt example file **<** system-wide config **<** user-wide config

.. note:: `Optimization` entries are required, `Dynamic` entry is optional.

"""
from __future__ import absolute_import

import argparse
from collections import defaultdict
from copy import deepcopy
import logging
import os
import socket
import textwrap

import six
import yaml

import metaopt


# Define type of arbitrary nested defaultdicts
def nesteddict():
    """Extend defaultdict to arbitrary nested levels."""
    return defaultdict(nesteddict)


log = logging.getLogger(__name__)

################################################################################
#                 Default Settings and Environmental Variables                 #
################################################################################

# Default settings for command line arguments (option, description)
DEF_CMD_MAX_TRIALS = (-1, 'inf/until preempted')
DEF_CMD_POOL_SIZE = (10, str(10))

DEF_CONFIG_FILES_PATHS = [
    os.path.join(metaopt.DIRS.site_data_dir, 'moptconfig.yaml.example'),
    os.path.join(metaopt.DIRS.site_config_dir, 'moptconfig.yaml'),
    os.path.join(metaopt.DIRS.user_config_dir, 'moptconfig.yaml')
    ]

# list containing tuples of
# (environmental variable names, configuration keys, default values)
ENV_VARS_DB = [
    ('METAOPT_DB_NAME', 'name', 'MetaOpt'),
    ('METAOPT_DB_TYPE', 'type', 'MongoDB'),
    ('METAOPT_DB_ADDRESS', 'address', socket.gethostbyname(socket.gethostname()))
    ]

# TODO: Default resource from environmental (localhost)

# dictionary describing lists of environmental tuples (e.g. `ENV_VARS_DB`)
# by a 'key' to be used in the experiment's configuration dict
ENV_VARS = dict(
    database=ENV_VARS_DB
    )

################################################################################
#                           Input Parsing Functions                            #
################################################################################


def mopt_args(description):
    """Get options from command line arguments.

    :param description: string description of ``mopt`` executable

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(description))

    parser.add_argument(
        '-V', '--version',
        action='version', version='metaopt ' + metaopt.__version__)

    parser.add_argument(
        '-v', '--verbose',
        action='count', default=0,
        help="logging levels of information about the process (-v: INFO. -vv: DEBUG)")

    moptgroup = parser.add_argument_group(
        "MetaOpt arguments (optional)",
        description="These arguments determine mopt's behaviour")

    moptgroup.add_argument(
        '-e', '--exp-name',
        type=str, metavar='stringID',
        help="experiment's unique name; "
             "use an existing name to resume an experiment "
             "(default: None - specified either here or in a config)")

    moptgroup.add_argument(
        '--max-trials', type=int, metavar='#',
        help="number of jobs/trials to be completed "
             "(default: %s)" % DEF_CMD_MAX_TRIALS[1])

    moptgroup.add_argument(
        "--pool-size", type=int, metavar='#',
        help="number of concurrent workers to evaluate candidate samples "
             "(default: %s)" % DEF_CMD_POOL_SIZE[1])

    moptgroup.add_argument(
        '-mc', '--moptconfig',
        type=argparse.FileType('r'), metavar='path-to-config',
        help="user provided mopt configuration file")

    usergroup = parser.add_argument_group(
        "User script related arguments",
        description="These arguments determine user's script behaviour "
                    "and they can serve as metaopt's parameter declaration.")

    usergroup.add_argument(
        '-c', '--userconfig',
        type=str, metavar='path-to-config',
        help="your script's configuration file (yaml, json, ini, ...anything)")

    usergroup.add_argument(
        'userscript', type=str, metavar='path-to-script',
        help="your experiment's script")

    usergroup.add_argument(
        'userargs', nargs=argparse.REMAINDER, metavar='...',
        help="command line arguments to your script (if any)")

    args = vars(parser.parse_args())  # convert to dict
    # Explicitly add metaopt's version as experiment's metadata
    args['version'] = metaopt.__version__
    moptfile = args.pop('moptconfig')
    config = dict()
    if moptfile:
        config = yaml.safe_load(moptfile)

    if args['verbose'] == 1:
        logging.basicConfig(level=logging.INFO)
    elif args['verbose'] == 2:
        logging.basicConfig(level=logging.DEBUG)

    return args, config


def default_options():
    """Create a nesteddict with options from the default configuration files.

    Respect precedence from application's default, to system's and
    user's.

    .. seealso:: :const:`DEF_CONFIG_FILES_PATHS`

    """
    defcfg = nesteddict()

    # get some defaults
    defcfg['exp_name'] = None
    defcfg['max_trials'] = DEF_CMD_MAX_TRIALS[0]
    defcfg['pool_size'] = DEF_CMD_POOL_SIZE[0]

    # get default options for some managerial variables (see :const:`ENV_VARS`)
    for signif, evars in six.iteritems(ENV_VARS):
        for _, key, default_value in evars:
            defcfg[signif][key] = default_value

    # fetch options from default configuration files
    for cfgpath in DEF_CONFIG_FILES_PATHS:
        try:
            with open(cfgpath) as f:
                cfg = yaml.safe_load(f)
                if cfg is None:
                    continue
                # implies that yaml must be in dict form
                for k, v in six.iteritems(cfg):
                    if k in ENV_VARS:
                        for vk, vv in six.iteritems(v):
                            defcfg[k][vk] = vv
                    else:
                        if k != 'exp_name':
                            defcfg[k] = v
        except IOError as e:  # default file could not be found
            log.debug(e)
        except AttributeError as e:
            log.warning("Problem parsing file: %s", cfgpath)
            log.warning(e)

    return defcfg


def env_vars(config):
    """Fetch environmental variables related to metaopt's managerial data.

    :type config: :func:`nesteddict`

    """
    newcfg = deepcopy(config)
    for signif, evars in six.iteritems(ENV_VARS):
        for var_name, key, _ in evars:
            value = os.getenv(var_name)
            if value is not None:
                newcfg[signif][key] = value
    return newcfg


def mopt_config(config, dbconfig, cmdconfig, cmdargs):
    """
    Finalize mopt configuration.

       * **name**: Experiment**s name. If you provide a past experiment**s name,
         then that experiment will be resumed. This means that its history of
         trials will be reused, along with any configurations logged in the
         database which are not overwritten by current call to `mopt` script.
         (default: <username>_<start datetime>)
       * **max_trials**: Maximum number of trial evaluations to be computed
         (required as a cmd line argument or a moptconfig parameter)
       * **pool_size**: Number of workers evaluating in parallel asychronously
         (default: 10 @ default resource). Can be a dict of the form:
         {resource_alias: subpool_size}
       * **database**: (db_opts)
       * **resources**: {resource_alias: (entry_address, scheduler, scheduler_ops)}
         (optional)
       * **optimizer**: {optimizer module name : method-specific configuration}
       * **dynamic**: {dynamic module name : method-specific configuration}

       .. seealso:: Method-specific configurations reside in `/config`

    """
    expconfig = deepcopy(config)

    for cfg in (dbconfig, cmdconfig):
        for k, v in six.iteritems(cfg):
            if k in ENV_VARS:
                for vk, vv in six.iteritems(v):
                    expconfig[k][vk] = vv
            else:
                expconfig[k] = v

    for k, v in six.iteritems(cmdargs):
        if v is not None:
            expconfig[k] = v

    return expconfig
