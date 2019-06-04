# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.exception` -- Custom exceptions for Oríon
================================================================

.. module:: exception
   :platform: Unix
   :synopsis: Custom exceptions.
"""


class NoConfigurationError(Exception):
    """Raise when commandline configuration is empty."""

    pass
