#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.insert` -- Module to insert new trials
================================================================

.. module:: insert
   :platform: Unix
   :synopsis: Insert creates new trials for a given experiment with fixed values

"""
import logging

from orion.core.cli import resolve_config
from orion.core.worker.experiment import Experiment

log = logging.getLogger(__name__)


def execute(cmdargs, cmdconfig):
    points = cmdargs.pop('user_args')


