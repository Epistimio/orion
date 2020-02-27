# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.exception` -- Custom exceptions for Oríon
================================================================

.. module:: exception
   :platform: Unix
   :synopsis: Custom exceptions.
"""


NO_CONFIGURATION_FOUND = """\
No commandline configuration found for new experiment.
"""


class NoConfigurationError(Exception):
    """Raise when commandline configuration is empty."""

    def __init__(self, message=NO_CONFIGURATION_FOUND):
        super().__init__(message)


class CheckError(Exception):
    """Raise when a check has failed."""

    pass


class RaceCondition(Exception):
    """Raise when a race condition occured."""

    pass


MISSING_RESULT_FILE = """
Cannot parse result file.

Make sure to report results in file `$ORION_RESULTS_PATH`.
This can be done with `orion.client.cli.report_objective()`.
"""


class MissingResultFile(Exception):
    """Raise when no result file (or empty) at end of trial execution."""

    def __init__(self, message=MISSING_RESULT_FILE):
        super().__init__(message)
