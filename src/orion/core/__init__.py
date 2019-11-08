# -*- coding: utf-8 -*-
"""
Oríon is an asynchronous distributed framework for black-box function optimization.

Its purpose is to serve as a hyperparameter optimizer for
machine learning models and training, as well as a flexible experimentation
platform for large scale asynchronous optimization procedures.

It has been designed firstly to disrupt a user's workflow at minimum, allowing
fast and efficient hyperparameter tuning, and secondly to provide secondary APIs
for more advanced features, such as dynamically reporting validation scores on
training time for automatic early stopping or on-the-fly reconfiguration.

Start by having a look here: https://github.com/epistimio/orion
"""
import logging
import os
import socket

from appdirs import AppDirs

from orion.core.io.config import Configuration
from ._version import get_versions


logger = logging.getLogger(__name__)


VERSIONS = get_versions()
del get_versions

__descr__ = 'Asynchronous [black-box] Optimization'
__version__ = VERSIONS['version']
__license__ = 'BSD-3-Clause'
__author__ = u'Epistímio'
__author_short__ = u'Epistímio'
__author_email__ = 'xavier.bouthillier@umontreal.ca'
__copyright__ = u'2017-2019, Epistímio'
__url__ = 'https://github.com/epistimio/orion'

DIRS = AppDirs(__name__, __author_short__)
del AppDirs

DEF_CONFIG_FILES_PATHS = [
    os.path.join(DIRS.site_data_dir, 'orion_config.yaml.example'),
    os.path.join(DIRS.site_config_dir, 'orion_config.yaml'),
    os.path.join(DIRS.user_config_dir, 'orion_config.yaml')
    ]


def define_config():
    """Create and define the fields of the configuration object."""
    config = Configuration()
    define_storage_config(config)
    define_experiment_config(config)
    define_worker_config(config)
    define_evc_config(config)

    config.add_option(
        'user_script_config', option_type=str, default='config')

    return config


def define_storage_config(config):
    """Create and define the fields of the storage configuration."""
    storage_config = Configuration()

    storage_config.add_option(
        'type', option_type=str, default='legacy', env_var='ORION_STORAGE_TYPE')

    config.storage = storage_config

    define_database_config(config.storage)
    # Backward compatibility, should be removed in v0.2.0
    config.database = config.storage.database


def define_database_config(config):
    """Create and define the fields of the database configuration."""
    database_config = Configuration()

    try:
        default_host = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        default_host = 'localhost'

    database_config.add_option(
        'name', option_type=str, default='orion', env_var='ORION_DB_NAME')
    database_config.add_option(
        'type', option_type=str, default='MongoDB', env_var='ORION_DB_TYPE')
    database_config.add_option(
        'host', option_type=str, default=default_host, env_var='ORION_DB_ADDRESS')
    database_config.add_option(
        'port', option_type=int, default=27017, env_var='ORION_DB_PORT')

    config.database = database_config


def define_experiment_config(config):
    """Create and define the fields of generic experiment configuration."""
    experiment_config = Configuration()

    experiment_config.add_option(
        'pool_size', option_type=int, default=1)

    experiment_config.add_option(
        'max_trials', option_type=int, default=int(10e8))

    experiment_config.add_option(
        'worker_trials', option_type=int, default=int(10e8))

    experiment_config.add_option(
        'working_dir', option_type=str, default='')

    experiment_config.add_option(
        'algorithms', option_type=dict, default={'random': {'seed': None}})

    experiment_config.add_option(
        'strategy', option_type=dict, default={'MaxParallelStrategy': {}})

    config.experiment = experiment_config


def define_worker_config(config):
    """Create and define the fields of the worker configuration."""
    worker_config = Configuration()

    worker_config.add_option(
        'heartbeat', option_type=int, default=120)
    worker_config.add_option(
        'max_broken', option_type=int, default=3)
    worker_config.add_option(
        'max_idle_time', option_type=int, default=60)

    config.worker = worker_config


def define_evc_config(config):
    """Create and define the fields of the evc configuration."""
    evc_config = Configuration()

    evc_config.add_option(
        'auto_resolution', option_type=bool, default=True)
    evc_config.add_option(
        'manual_resolution', option_type=bool, default=False)
    evc_config.add_option(
        'non_monitored_arguments', option_type=list, default=[])
    evc_config.add_option(
        'ignore_code_changes', option_type=bool, default=False)
    evc_config.add_option(
        'algorithm_change', option_type=bool, default=False)
    evc_config.add_option(
        'code_change_type', option_type=str, default='break')
    evc_config.add_option(
        'cli_change_type', option_type=str, default='break')
    evc_config.add_option(
        'config_change_type', option_type=str, default='break')

    config.evc = evc_config


def build_config():
    """Define the config and fill it based on global configuration files."""
    config = define_config()
    for file_path in DEF_CONFIG_FILES_PATHS:
        if not os.path.exists(file_path):
            logger.debug('Config file not found: %s', file_path)
            continue

        config.load_yaml(file_path)

    return config


config = build_config()
