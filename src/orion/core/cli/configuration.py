#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
:mod:`orion.core.utils.configuration` -- Singleton object working storing the configuration of orion for the program duration

.. module:: cli
    :platform: Unix
    :synopsis: Helper object for configuration of Orion

"""

import logging
import orion

from orion.core.utils import SingletonType

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
    ('METAOPT_DB_NAME', 'name', 'Orion'),
    ('METAOPT_DB_TYPE', 'type', 'MongoDB'),
    ('METAOPT_DB_ADDRESS', 'host', socket.gethostbyname(socket.gethostname()))
    ]

# TODO: Default resource from environmental (localhost)

# dictionary describing lists of environmental tuples (e.g. `ENV_VARS_DB`)
# by a 'key' to be used in the experiment's configuration dict
ENV_VARS = dict(
    database=ENV_VARS_DB
    )

def is_exe(path):
    """Test whether `path` describes an executable file"""
    return os.path.isfile(path) and os.access(path, os.X_OK)

class Configuration(metaclass=SingletonType)
    def __init__(self, cmdargs)
        self.config = vars(cmdargs)

        verbose = self.config.pop('verbose', 0)
        if verbose == 1:
            logging.basicConfig(level=logging.INFO)
        elif verbose == 2:
            logging.basicConfig(level=logging.DEBUG)

        self.config['metadata'] = dict()
        self.config['metadata']['orion_version'] = orion.core.__version__
        log.debug("Using orion version %s", self.config['metadata']['orion_version'])

        orion_file = self.config.pop('config')

        if orion_file:
            log.debug("Found orion configuration file at: %s", os.path.abspath(orion_file.name))
            config = yaml.safe_load(orion_file)

        user_script = config.pop('user_script', "")
        abs_user_script = os.path.abspath(user_script)

        if is_exe(abs_user_script):
            user_script = abs_user_script

        self.config['metadata']['user_script'] = user_script
        self.config['metadata']['user_args'] = self.config.pop('user_args')
        

