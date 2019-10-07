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
