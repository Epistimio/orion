"""
:mod:`orion.serving.parameters` -- Common code for verifying query parameters
=============================================================================

.. module:: parameters
   :platform: Unix
   :synopsis: Common code related to query parameters verification
"""


def verify_query_parameters(parameters: dict, supported_parameters: list):
    """
    Verifies that the parameters given in the input dictionary are all supported.

    Parameters
    ----------
    parameters
        The dictionary of parameters to verify in the format ``parameter_name:value``.

    supported_parameters
        The list of parameter names that are supported.

    Returns
    -------
    An error message if a parameter is invalid; otherwise None.
    """

    for key in parameters:
        if key not in supported_parameters:
            return _compose_error_message(key, list(supported_parameters.keys()))


def _compose_error_message(key: str, supported_parameters: list):
    """Creates the error message depending on the number of supported parameters available."""
    error_message = f"Parameter '{key}' is not supported. Expected "

    if len(supported_parameters) > 1:
        supported_parameters.sort()
        error_message += f"one of {supported_parameters}."
    else:
        error_message += f"parameter '{supported_parameters[0]}'."

    return error_message
