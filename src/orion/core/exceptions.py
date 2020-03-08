# -*- coding: utf-8 -*-
"""
:mod:`orion.core.exceptions` -- Orion front facing exceptions
=============================================================

.. module:: exceptions
   :platform: Unix
   :synopsis: define public exceptions

"""


class SampleTimeout(Exception):
    """Raised when the algorithm is not able to sample new unique points in time"""

    pass


class WaitingForTrials(Exception):
    """Raised when the algorithm needs to wait for some trials to complete before it can suggest new ones"""

    pass


class BrokenExperiment(Exception):
    """Raised when a trial has been tried too many times without success"""

    pass
