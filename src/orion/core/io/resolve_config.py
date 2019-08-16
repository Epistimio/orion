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

Default files are given as a list at :const:`orion.core.DEF_CONFIG_FILES_PATHS` and a
precedence is respected when building the settings dictionary:

 * default orion example file **<** system-wide config **<** user-wide config

.. note:: `Optimization` entries are required, `Dynamic` entry is optional.

"""
import errno
import getpass
import hashlib
import logging
import os

import git
from numpy import inf as infinity
import yaml

import orion
import orion.core
from orion.core import config


def is_exe(path):
    """Test whether `path` describes an executable file."""
    return os.path.isfile(path) and os.access(path, os.X_OK)


log = logging.getLogger(__name__)

################################################################################
#                 Default Settings and Environmental Variables                 #
################################################################################

# Default settings for command line arguments (option, description)
DEF_CMD_MAX_TRIALS = (infinity, 'inf/until preempted')
DEF_CMD_WORKER_TRIALS = (infinity, 'inf/until preempted')
DEF_CMD_POOL_SIZE = (1, str(1))

# list containing tuples of
# (environmental variable names, configuration keys, default values)
ENV_VARS_DB = [
    ('ORION_DB_NAME', 'name'),
    ('ORION_DB_TYPE', 'type'),
    ('ORION_DB_ADDRESS', 'host'),
    ('ORION_DB_PORT', 'port')
    ]

# TODO: Default resource from environmental (localhost)

# dictionary describing lists of environmental tuples (e.g. `ENV_VARS_DB`)
# by a 'key' to be used in the experiment's configuration dict
ENV_VARS = dict(
    database=ENV_VARS_DB
    )


def fetch_config(args):
    """Return the config inside the .yaml file if present."""
    orion_file = args.get('config')
    config = dict()
    if orion_file:
        log.debug("Found orion configuration file at: %s", os.path.abspath(orion_file.name))
        orion_file.seek(0)
        config = yaml.safe_load(orion_file)

    return config


def fetch_default_options():
    """Create a dict with options from the default configuration files.

    Respect precedence from application's default, to system's and
    user's.

    .. seealso:: :const:`orion.core.DEF_CONFIG_FILES_PATHS`

    """
    default_config = dict()

    # get some defaults
    default_config['name'] = None
    default_config['user'] = getpass.getuser()
    default_config['max_trials'] = DEF_CMD_MAX_TRIALS[0]
    default_config['worker_trials'] = DEF_CMD_WORKER_TRIALS[0]
    default_config['pool_size'] = DEF_CMD_POOL_SIZE[0]
    default_config['algorithms'] = 'random'

    # get default options for some managerial variables (see :const:`ENV_VARS`)
    for signifier, env_vars in ENV_VARS.items():
        default_config[signifier] = {}
        for _, key in env_vars:
            default_config[signifier][key] = config[signifier][key]

    # fetch options from default configuration files
    for configpath in orion.core.DEF_CONFIG_FILES_PATHS:
        try:
            with open(configpath) as f:
                cfg = yaml.safe_load(f)
                if cfg is None:
                    continue
                # implies that yaml must be in dict form
                for k, v in cfg.items():
                    if k in ENV_VARS:
                        default_config[k] = {}
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


def fetch_env_vars():
    """Fetch environmental variables related to orion's managerial data."""
    env_vars = {}

    for signif, evars in ENV_VARS.items():
        env_vars[signif] = {}

        for var_name, key in evars:
            value = os.getenv(var_name)

            if value is not None:
                env_vars[signif][key] = value

    return env_vars


def fetch_metadata(cmdargs):
    """Infer rest information about the process + versioning"""
    metadata = {}

    metadata['orion_version'] = orion.core.__version__

    # Move 'user_script' and 'user_args' to 'metadata' key
    user_args = cmdargs.get('user_args', [])

    # Trailing white space are catched by argparse as an empty argument
    if len(user_args) == 1 and user_args[0] == '':
        user_args = []

    user_script = user_args[0] if user_args else None

    if user_script:
        abs_user_script = os.path.abspath(user_script)
        if is_exe(abs_user_script):
            user_script = abs_user_script

    if user_script and not os.path.exists(user_script):
        raise OSError(errno.ENOENT, "The path specified for the script does not exist", user_script)

    if user_script:
        metadata['user_script'] = user_script
        metadata['VCS'] = infer_versioning_metadata(metadata['user_script'])

    if user_args:
        metadata['user_args'] = user_args[1:]

    metadata['user'] = getpass.getuser()
    return metadata


def merge_configs(*configs):
    """Merge configuration dictionnaries following the given hierarchy

    Suppose function is called as merge_configs(A, B, C). Then any pair (key, value) in C would
    overwrite any previous value from A or B. Same apply for B over A.

    If for some pair (key, value), the value is a dictionary, then it will either overwrite previous
    value if it was not also a directory, or it will be merged following
    `merge_configs(old_value, new_value)`.

    .. warning:

        Redefinition of subdictionaries may lead to confusing results because merges do not remove
        data.

        If for instance, we have {'a': {'b': 1, 'c': 2}} and we would like to update `'a'` such that
        it only have `{'c': 3}`, it won't work with {'a': {'c': 3}}.

        merge_configs({'a': {'b': 1, 'c': 2}}, {'a': {'c': 3}}) -> {'a': {'b': 1, 'c': 3}}

    Examples
    --------
    .. code-block:: python
        :linenos:

        a = {'a': 1, 'b': {'c': 2}}
        b = {'b': {'c': 3}}
        c = {'b': {'c': {'d': 4}}}

        m = resolve_config.merge_configs(a, b, c)

        assert m == {'a': 1, 'b': {'c': {'d': 4}}}

        a = {'a': 1, 'b': {'c': 2, 'd': 3}}
        b = {'b': {'c': 4}}
        c = {'b': {'c': {'e': 5}}}

        m = resolve_config.merge_configs(a, b, c)

        assert m == {'a': 1, 'b': {'c': {'e': 5}, 'd': 3}}

    """
    merged_config = configs[0]

    for config_i in configs[1:]:
        for key, value in config_i.items():
            if isinstance(value, dict) and isinstance(merged_config.get(key), dict):
                merged_config[key] = merge_configs(merged_config[key], value)
            elif value is not None:
                merged_config[key] = value

    return merged_config


def fetch_user_repo(user_script):
    """Fetch the GIT repo and its root path given user's script."""
    dir_path = os.path.dirname(os.path.abspath(user_script))
    try:
        git_repo = git.Repo(dir_path, search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        git_repo = None
        logging.warning('Script %s is not in a git repository. Code modification '
                        'won\'t be detected.', os.path.abspath(user_script))
    return git_repo


def infer_versioning_metadata(user_script):
    """
    Infer information about user's script versioning if available.
    Fills the following information in VCS:

    `is_dirty` shows whether the git repo is at a clean state.
    `HEAD_sha` gives the hash of head of the repo.
    `active_branch` shows the active branch of the repo.
    `diff_sha` shows the hash of the diff in the repo.

    :returns: the `VCS` but filled with above info.

    """
    git_repo = fetch_user_repo(user_script)
    if not git_repo:
        return {}
    vcs = {}
    vcs['type'] = 'git'
    vcs['is_dirty'] = git_repo.is_dirty()
    vcs['HEAD_sha'] = git_repo.head.object.hexsha
    if git_repo.head.is_detached:
        vcs['active_branch'] = None
    else:
        vcs['active_branch'] = git_repo.active_branch.name
    # The 'diff' of the current version from the latest commit
    diff = git_repo.git.diff(git_repo.head.commit.tree).encode('utf-8')
    diff_sha = hashlib.sha256(diff).hexdigest()
    vcs['diff_sha'] = diff_sha
    return vcs
