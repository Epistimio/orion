# -*- coding: utf-8 -*-
"""
:mod:`orion.core.exceptions` -- Orion front facing exceptions
=============================================================

.. module:: exceptions
   :platform: Unix
   :synopsis: define public exceptions

"""


class SampleTimeout(Exception):
    """Raises when the algorithm is not able to sample new unique points in time"""

    pass


class WaitingForTrials(Exception):
    """Raised when no trials could be reserved after multiple retries"""

    pass


class BrokenExperiment(Exception):
    """Raised a trial has been tried too many times without success"""

    pass
