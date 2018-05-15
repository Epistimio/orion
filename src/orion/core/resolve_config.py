# -*- coding: utf-8 -*-
"""
:mod:`orion.core.resolve_config` -- Configuration parsing and resolving
=======================================================================

.. module:: resolve_config
   :platform: Unix
   :synopsis: How does orion resolve configuration settings?

How:

 - Experiment name resolves like this:
    * cmd-arg **>** cmd-provided orion_config **>** REQUIRED (no default is given)

 - Database options resolve with the following precedence (high to low):
    * cmd-provided orion_config **>** env vars **>** default files **>** defaults

.. seealso:: :const:`ENV_VARS`, :const:`ENV_VARS_DB`


 - All other managerial, `Optimization` or `Dynamic` options resolve like this:

    * cmd-args **>** cmd-provided orion_config **>** database (if experiment name
      can be found) **>** default files

Default files are given as a list at :const:`DEF_CONFIG_FILES_PATHS` and a
precedence is respected when building the settings dictionary:

 * default orion example file **<** system-wide config **<** user-wide config

.. note:: `Optimization` entries are required, `Dynamic` entry is optional.

"""
import argparse
from collections import defaultdict
from copy import deepcopy
import logging
import os
import socket
import textwrap

from numpy import inf as infinity
import yaml

import orion


# Define type of arbitrary nested defaultdicts
def nesteddict():
    """Extend defaultdict to arbitrary nested levels."""
    return defaultdict(nesteddict)


def is_exe(path):
    """Test whether `path` describes an executable file."""
    return os.path.isfile(path) and os.access(path, os.X_OK)


log = logging.getLogger(__name__)

################################################################################
#                 Default Settings and Environmental Variables                 #
################################################################################

# Default settings for command line arguments (option, description)
DEF_CMD_MAX_TRIALS = (infinity, 'inf/until preempted')
DEF_CMD_POOL_SIZE = (10, str(10))

DEF_CONFIG_FILES_PATHS = [
    os.path.join(orion.core.DIRS.site_data_dir, 'orion_config.yaml.example'),
    os.path.join(orion.core.DIRS.site_config_dir, 'orion_config.yaml'),
    os.path.join(orion.core.DIRS.user_config_dir, 'orion_config.yaml')
    ]

# list containing tuples of
# (environmental variable names, configuration keys, default values)
ENV_VARS_DB = [
    ('ORION_DB_NAME', 'name', 'orion'),
    ('ORION_DB_TYPE', 'type', 'MongoDB'),
    ('ORION_DB_ADDRESS', 'host', socket.gethostbyname(socket.gethostname()))
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


def fetch_orion_args(description):
    """Get options from command line arguments.

    :param description: string description of ``orion`` executable

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(description))

    parser.add_argument(
        '-V', '--version',
        action='version', version='orion ' + orion.core.__version__)

    parser.add_argument(
        '-v', '--verbose',
        action='count', default=0,
        help="logging levels of information about the process (-v: INFO. -vv: DEBUG)")

    orion_group = parser.add_argument_group(
        u"Oríon arguments (optional)",
        description="These arguments determine orion's behaviour")

    orion_group.add_argument(
        '-n', '--name',
        type=str, metavar='stringID',
        help="experiment's unique name; "
             "use an existing name to resume an experiment "
             "(default: None - specified either here or in a config)")

    orion_group.add_argument(
        '--max-trials', type=int, metavar='#',
        help="number of jobs/trials to be completed "
             "(default: %s)" % DEF_CMD_MAX_TRIALS[1])

    orion_group.add_argument(
        "--pool-size", type=int, metavar='#',
        help="number of concurrent workers to evaluate candidate samples "
             "(default: %s)" % DEF_CMD_POOL_SIZE[1])

    orion_group.add_argument(
        '-c', '--config',
        type=argparse.FileType('r'), metavar='path-to-config',
        help="user provided orion configuration file")

    usergroup = parser.add_argument_group(
        "User script related arguments",
        description="These arguments determine user's script behaviour "
                    "and they can serve as orion's parameter declaration.")

    usergroup.add_argument(
        'user_script', type=str, metavar='path-to-script',
        help="your experiment's script")

    usergroup.add_argument(
        'user_args', nargs=argparse.REMAINDER, metavar='...',
        help="Command line arguments to your script (if any). A configuration "
             "file intended to be used with 'userscript' must be given as a path "
             "in the **first positional** argument OR using `--config=<path>` "
             "keyword argument.")

    args = vars(parser.parse_args())  # convert to dict

    verbose = args.pop('verbose')
    if verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    # Explicitly add orion's version as experiment's metadata
    args['metadata'] = dict()
    args['metadata']['orion_version'] = orion.core.__version__
    log.debug("Using orion version %s", args['metadata']['orion_version'])

    orion_file = args.pop('config')
    config = dict()
    if orion_file:
        log.debug("Found orion configuration file at: %s", os.path.abspath(orion_file.name))
        config = yaml.safe_load(orion_file)

    # Move 'user_script' and 'user_args' to 'metadata' key
    user_script = args.pop('user_script')
    abs_user_script = os.path.abspath(user_script)
    if is_exe(abs_user_script):
        user_script = abs_user_script
    args['metadata']['user_script'] = user_script
    args['metadata']['user_args'] = args.pop('user_args')
    log.debug("Problem definition: %s %s", args['metadata']['user_script'],
              ' '.join(args['metadata']['user_args']))

    return args, config


def fetch_default_options():
    """Create a nesteddict with options from the default configuration files.

    Respect precedence from application's default, to system's and
    user's.

    .. seealso:: :const:`DEF_CONFIG_FILES_PATHS`

    """
    default_config = nesteddict()

    # get some defaults
    default_config['name'] = None
    default_config['max_trials'] = DEF_CMD_MAX_TRIALS[0]
    default_config['pool_size'] = DEF_CMD_POOL_SIZE[0]
    default_config['algorithms'] = 'random'

    # get default options for some managerial variables (see :const:`ENV_VARS`)
    for signifier, env_vars in ENV_VARS.items():
        for _, key, default_value in env_vars:
            default_config[signifier][key] = default_value

    # fetch options from default configuration files
    for configpath in DEF_CONFIG_FILES_PATHS:
        try:
            with open(configpath) as f:
                cfg = yaml.safe_load(f)
                if cfg is None:
                    continue
                # implies that yaml must be in dict form
                for k, v in cfg.items():
                    if k in ENV_VARS:
                        for vk, vv in v.items():
                            default_config[k][vk] = vv
                    else:
                        if k != 'name':
                            default_config[k] = v
        except IOError as e:  # default file could not be found
            log.debug(e)
        except AttributeError as e:
            log.warning("Problem parsing file: %s", configpath)
            log.warning(e)

    return default_config


def merge_env_vars(config):
    """Fetch environmental variables related to orion's managerial data.

    :type config: :func:`nesteddict`

    """
    newcfg = deepcopy(config)
    for signif, evars in ENV_VARS.items():
        for var_name, key, _ in evars:
            value = os.getenv(var_name)
            if value is not None:
                newcfg[signif][key] = value
    return newcfg


def merge_orion_config(config, dbconfig, cmdconfig, cmdargs):
    """
    Oríon Configuration
    -------------------

    name --  Experiment's name.

       If you provide a past experiment's name,
       then that experiment will be resumed. This means that its history of
       trials will be reused, along with any configurations logged in the
       database which are not overwritten by current call to `orion` script.

    max_trials -- Maximum number of trial evaluations to be computed

       (required as a cmd line argument or a orion_config parameter)

    pool_size -- Number of workers evaluating in parallel asychronously

       (default: 10 @ default resource). Can be a dict of the form:
       {resource_alias: subpool_size}

    database -- dict with keys: 'type', 'name', 'host', 'port', 'username', 'password'

    resources -- {resource_alias: (entry_address, scheduler, scheduler_ops)} (optional)

    algorithm -- {optimizer module name : method-specific configuration}

    .. seealso:: Method-specific configurations reside in `/config`

    """
    expconfig = deepcopy(config)

    for cfg in (dbconfig, cmdconfig):
        for k, v in cfg.items():
            if k in ENV_VARS:
                for vk, vv in v.items():
                    expconfig[k][vk] = vv
            elif v is not None:
                expconfig[k] = v

    for k, v in cmdargs.items():
        if v is not None:
            if k == 'metadata':
                for vk, vv in v.items():
                    expconfig[k][vk] = vv
            else:
                expconfig[k] = v

    return expconfig
