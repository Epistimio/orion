"""
:mod:`orion.serving.parameters` -- Common code for verifying query parameters
=============================================================================

.. module:: parameters
   :platform: Unix
   :synopsis: Common code related to query parameters verification
"""


def verify_parameter_type(parameter_value: str, expected_type: str) -> bool:
    """Verifies that the parameter value is interpretable as the given expected type."""
    if expected_type == 'str':
        return True
    elif expected_type == 'bool':
        return parameter_value.lower() == 'false' or parameter_value.lower() == 'true'
    elif expected_type == 'int':
        try:
            int(parameter_value)
        except ValueError:
            return False
        return True
    else:
        return False


def verify_query_parameters(parameters: dict, supported_parameters: dict):
    """
    Verifies that the parameters given in the input dictionary are all supported.

    Parameters
    ----------
    parameters
        The dictionary of parameters to verify in the format ``parameter_name:value``.

    supported_parameters
        The dictionary of parameter names and types that are supported.
        The format is ``parameter_name:type``

    Returns
    -------
    An error message if a parameter is invalid; otherwise None.
    """

    for key in parameters:
        if key not in supported_parameters:
            return _compose_error_message(key, list(supported_parameters.keys()))

        expected_type = supported_parameters[key]
        if not verify_parameter_type(parameters.get(key), expected_type):
            return f"Incorrect type for parameter '{key}'. Expected type '{expected_type}'."


def _compose_error_message(key: str, supported_parameters: list):
    """Creates the error message depending on the number of supported parameters available."""
    error_message = f"Parameter '{key}' is not supported. Expected "

    if len(supported_parameters) > 1:
        supported_parameters.sort()
        error_message += f"one of {supported_parameters}."
    else:
        error_message += f"parameter '{supported_parameters[0]}'."

    return error_message
