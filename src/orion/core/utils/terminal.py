#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.terminal` -- Helper functions for terminal i/o
=====================================================================

.. module:: setup
   :platform: Unix
   :synopsis: Helper functions for interaction with user on terminal
"""


def ask_question(question, default=None):
    """Ask a question to the user and receive an answer.

    Parameters
    ----------
    question: str
        The question to be asked.
    default: str
        The default value to use if the user enters nothing.

    Returns
    -------
    str
        The answer provided by the user.

    """
    if default is not None:
        question = question + " (default: {}) ".format(default)

    answer = input(question)

    if answer.strip() == "":
        return default

    return answer


def confirm_name(message, name, force=False):
    """Ask the user to confirm the name.

    Parameters
    ----------
    message: str
        The message to be printed.
    name: str
        The string that the user must enter.
    force: bool
        Override confirmation and return True. Default: False.

    Returns
    -------
    bool
        True if confirmed, False otherwise.

    """
    if force:
        print(message)
        print("FORCED")
        return True

    answer = input(message)

    return answer.strip() == name
