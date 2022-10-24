"""
Helper function for returning results from script
=================================================
"""
import os
import sys

from orion.core import config

IS_ORION_ON = False
_HAS_REPORTED_RESULTS = False
RESULTS_FILENAME = os.getenv("ORION_RESULTS_PATH", None)

if RESULTS_FILENAME and os.path.isfile(RESULTS_FILENAME):
    import json

    IS_ORION_ON = True

if RESULTS_FILENAME and not IS_ORION_ON:
    raise RuntimeWarning(
        f"Results file path ({RESULTS_FILENAME}) provided in environmental variable "
        "does not correspond to an existing file. "
    )


def interrupt_trial():
    """Send interrupt signal to Oríon worker"""
    sys.exit(config.worker.interrupt_signal_code)


def report_objective(objective, name="objective"):
    """Report only the objective at the end of execution

    To send more data (statistic, constraint, gradient), use ``report_results``.

    .. warning::

       To be called only once in order to report a final evaluation of a particular trial.

    .. warning::

       Oríon is only minimizing. Make sure to report a metric that you seek to minimize.

    .. note::

       In case that user's script is not running in a orion's context,
       this function will act as a Python `print` function.

    Parameters
    ----------
    objective: float
        Objective the return to Oríon for the current trial.
    name: str, optional
        Name of the objective. Default is 'objective'.

    """
    report_results([dict(name=name, type="objective", value=objective)])


def report_bad_trial(objective=1e10, name="objective", data=None):
    """Report a bad trial with large objective to Oríon.

    This is especially useful if some parameter values lead to exceptions such as out of memory.
    Reporting a large objective from such trials will push algorithms towards valid
    configurations.

    .. warning::

       To be called only once in order to report a final evaluation of a particular trial.

    .. warning::

       Oríon is only minimizing. Make sure to report a metric that you seek to minimize.

    .. note::

       In case that user's script is not running in a orion's context,
       this function will act as a Python `print` function.

    Parameters
    ----------
    objective: float
        Objective the return to Oríon for the current trial. The default objective is 1e10.
        This may not be valid for some metrics and this value should be overridden accordingly. In
        the case of error rates for instance, the value should be 1.0.
    name: str, optional
        Name of the objective. Default is 'objective'.
    data: list of dict, optional
        A list of dictionary representing the results in the form
        dict(name=result_name, type='statistic', value=0). The types supported are
        'contraint', 'gradient' and 'statistic'.

    """
    if data is None:
        data = []
    report_results([dict(name=name, type="objective", value=objective)] + data)


def report_results(data):
    """Facilitate the reporting of results for a user's script acting as a
    black-box computation.

    .. warning::

       To be called only once in order to report a final evaluation of a particular trial.

    .. warning::

       Oríon is only minimizing. Make sure to report a metric that you seek to minimize.

    .. note::

       In case that user's script is not running in a orion's context,
       this function will act as a Python `print` function.

    Parameters
    ----------
    data: list of dict
        A list of dictionary representing the results in the form
        dict(name=result_name, type='statistic', value=0). The types supported are
        'objective', 'contraint', 'gradient' and 'statistic'. The list should contain at least
        one 'objective', which is the metric the algorithm will be minimizing.

    """
    global _HAS_REPORTED_RESULTS  # pylint:disable=global-statement
    if _HAS_REPORTED_RESULTS:
        raise RuntimeWarning("Has already reported evaluation results once.")
    if IS_ORION_ON:
        with open(RESULTS_FILENAME, "w", encoding="utf8") as results_file:
            json.dump(data, results_file)
    else:
        print(data)
    _HAS_REPORTED_RESULTS = True
