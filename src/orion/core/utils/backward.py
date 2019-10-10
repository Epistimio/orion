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


def populate_priors(metadata):
    """Compute parser state and priors based on user_args and populate metadata."""
    if 'user_args' not in metadata:
        return

    parser = OrionCmdlineParser(orion.core.config.user_script_config)
    parser.parse(metadata["user_args"])
    metadata["parser"] = parser.get_state_dict()
    metadata["priors"] = dict(parser.priors)


def db_is_outdated(database):
    """Return True if the database scheme is outdated."""
    deprecated_indices = [('name', 'metadata.user'), ('name', 'metadata.user', 'version'),
                          'name_1_metadata.user_1', 'name_1_metadata.user_1_version_1']

    index_information = database.index_information('experiments')
    return any(index in deprecated_indices for index in index_information.keys())
