# -*- coding: utf-8 -*-
"""
:mod:`metaopt.client` -- Helper function for returning results from script
==========================================================================

.. module:: client
   :platform: Unix
   :synopsis: Provides functions for communicating with `metaopt.core`.

"""
import os

IS_METAOPT_ON = False
RESULTS_FILENAME = os.getenv('METAOPT_RESULTS_PATH', None)
if RESULTS_FILENAME and os.path.isfile(RESULTS_FILENAME):
    import json
    IS_METAOPT_ON = True

if RESULTS_FILENAME and not IS_METAOPT_ON:
    raise RuntimeWarning("Results file path provided in environmental variable "
                         "does not correspond to an existing file.")

_HAS_REPORTED_RESULTS = False


def report_results(data):
    """Facilitate the reporting of results for a user's script acting as a
    black-box computation.

    :param data: A dictionary containing experimental results

    .. note:: To be called only once in order to report a final evaluation
       of a particular trial.

    .. note:: In case that user's script is not running in a metaopt's context,
       this function will act as a Python `print` function.

    .. note:: For your own good, this can be called **only once**.

    """
    global _HAS_REPORTED_RESULTS  # pylint:disable=global-statement
    if _HAS_REPORTED_RESULTS:
        raise RuntimeWarning("Has already reported evaluation results once.")
    if IS_METAOPT_ON:
        with open(RESULTS_FILENAME, 'w') as results_file:
            json.dump(data, results_file)
    else:
        print(data)
    _HAS_REPORTED_RESULTS = True
