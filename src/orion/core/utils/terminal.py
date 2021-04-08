#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions for terminal i/o
=================================
"""


def ask_question(question, default=None, choice=None, ignore_case=False):
    """Ask a question to the user and receive an answer.

    Parameters
    ----------
    question: str
        The question to be asked.
    default: str
        The default value to use if the user enters nothing.
    choice: list
        List of expected values to check user answer
    ignore_case: bool
        Used only if choice is provided. If True, ignore case when checking
        user answer against given choice.

    Returns
    -------
    str
        The answer provided by the user.

    """
    if choice is not None:
        if ignore_case:
            choice = [value.lower() for value in choice]
        question = question + " (choice: {})".format(", ".join(choice))

    if default is not None:
        question = question + " (default: {}) ".format(default)

    while True:
        answer = input(question)
        if answer.strip() == "":
            answer = default
            break
        if choice is None:
            break
        if answer in choice or (ignore_case and answer.lower() in choice):
            break
        print(
            "Unexpected value: {}. Must be one of: {}\n".format(
                answer, ", ".join(choice)
            )
        )

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
