#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.db.setup` -- Module running the setup command
===============================================================

.. module:: setup
   :platform: Unix
   :synopsis: Creates a configurarion file for the database.
"""

import logging
import os

import yaml

import orion.core

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Return the parser that needs to be used for this command"""
    setup_parser = parser.add_parser('setup', help='setup help')

    setup_parser.set_defaults(func=main)

    return setup_parser


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


# pylint: disable = unused-argument
def main(*args):
    """Build a configuration file."""
    default_file = orion.core.DEF_CONFIG_FILES_PATHS[-1]

    if os.path.exists(default_file):
        cancel = ''
        while cancel.strip().lower() not in ['y', 'n']:
            cancel = ask_question(
                "This will overwrite {}, do you want to proceed? (y/n) ".format(default_file), "n")

        if cancel.strip().lower() == 'n':
            return

    _type = ask_question("Enter the database type: ", "mongodb")
    name = ask_question("Enter the database name: ", "test")
    host = ask_question("Enter the database host: ", "localhost")

    config = {'database': {'type': _type, 'name': name, 'host': host}}

    print("Default configuration file will be saved at: ")
    print(default_file)

    dirs = '/'.join(default_file.split('/')[:-1])
    os.makedirs(dirs, exist_ok=True)

    with open(default_file, 'w') as output:
        yaml.dump(config, output, default_flow_style=False)
