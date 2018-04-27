#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli` -- Functions that define console scripts
================================================================

.. module:: cli
   :platform: Unix
   :synopsis: Helper functions to setup an experiment and execute it.

"""
import logging

from orion.core.cli import resolve_config
from orion.core.io.database import Database, DuplicateKeyError
from orion.core.worker import workon
from orion.core.worker.experiment import Experiment
from orion.client import manual
from importlib import import_module

log = logging.getLogger(__name__)

CLI_DOC_HEADER = """
orion:
  Orion cli script for asynchronous distributed optimization

"""


def main():
    """Entry point for `orion.core` functionality."""
    
    """Use `orion.core.resolve_config` to organize how configuration is built."""
    # Fetch experiment name, user's script path and command line arguments
    # Use `-h` option to show help
 
    cmdargs, cmdconfig = resolve_config.OrionArgsParser(CLI_DOC_HEADER)()
  
    #get the function used as default for the command and remove it from the current args
    command = cmdargs.pop('func')

    #import the corresponding module inside the cli folder
    module = __import__('orion.core.cli.' + command.__name__, fromlist=[''])

    #get the 'execute' function as a callable object
    execute = getattr(module, 'execute')

    execute(cmdargs, cmdconfig)

    return 0

if __name__ == "__main__":
    main()
