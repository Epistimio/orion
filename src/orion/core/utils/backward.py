#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.backward` -- Backward compatibility utils
================================================================

.. module:: info
   :platform: Unix
   :synopsis: Helper functions to support backward compatibility

"""

import orion.core
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser


def update_user_args(metadata):
    """Make sure user script is not removed from metadata"""
    if "user_script" in metadata and metadata["user_script"] not in metadata["user_args"]:
        metadata["user_args"] = [metadata["user_script"]] + metadata["user_args"]


def populate_priors(metadata):
    """Compute parser state and priors based on user_args and populate metadata."""
    if 'user_args' not in metadata:
        return

    update_user_args(metadata)

    parser = OrionCmdlineParser(orion.core.config.worker.user_script_config,
                                allow_non_existing_user_script=True)
    parser.parse(metadata["user_args"])
    metadata["parser"] = parser.get_state_dict()
    metadata["priors"] = dict(parser.priors)


def populate_space(config):
    """Add the space definition at the root of config."""
    populate_priors(config['metadata'])
    # Overwrite space to make sure to include changes from user_args
    if 'priors' in config['metadata']:
        config['space'] = config['metadata']['priors']


def db_is_outdated(database):
    """Return True if the database scheme is outdated."""
    deprecated_indices = [('name', 'metadata.user'), ('name', 'metadata.user', 'version'),
                          'name_1_metadata.user_1', 'name_1_metadata.user_1_version_1']

    index_information = database.index_information('experiments')
    return any(index in deprecated_indices for index in index_information.keys())


def update_db_config(config):
    """Merge DB config back into storage config"""
    config.setdefault('storage', orion.core.config.storage.to_dict())
    if 'database' in config:
        config['storage'] = {'type': 'legacy'}
        config['storage']['database'] = config.pop('database')
