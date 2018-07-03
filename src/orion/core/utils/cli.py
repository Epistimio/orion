# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.diff` -- Utilitary functions for command line
====================================================================

.. module:: experiment
   :platform: Unix
   :synopsis: Utilitary functions for operations related to command line

"""

import logging

log = logging.getLogger(__name__)


def any_type(value_as_string):
    """Convert a string to python object if applicable.

    This function is useful when user may provide a value which
    is not a string but is considered so by the command line interpreter and
    cannot be infered in advanced by the developer.
    """
    try:
        return int(value_as_string)
    except ValueError:
        log.debug("'%s' is not an int", value_as_string)

    try:
        return float(value_as_string)
    except ValueError:
        log.debug("'%s' is not a float", value_as_string)

    try:
        return eval(value_as_string)  # pylint:disable=eval-used
    except Exception:  # pylint:disable=broad-except
        return value_as_string
