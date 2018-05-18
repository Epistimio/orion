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
import logging
import os
import socket

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


def fetch_config(args):
    """Return the config inside the .yaml file if present."""
    orion_file = args.pop('config')
    config = dict()
    if orion_file:
        log.debug("Found orion configuration file at: %s", os.path.abspath(orion_file.name))
        config = yaml.safe_load(orion_file)

    return config


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


def merge_configs(*configs):
    raise NotImplementedError


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


def infer_versioning_metadata(existing_metadata):
    """Infer information about user's script versioning if available."""
    # VCS system
    # User repo's version
    # User repo's HEAD commit hash
    return existing_metadata
