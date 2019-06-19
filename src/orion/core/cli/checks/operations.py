#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.checks.operations` -- Operations stage for database checks
===============================================================================

.. module:: operations
    :platform: Unix
    :synopsis: Checks for the operations of a `Database` object.

"""

from orion.core.utils.decorators import register_check
from orion.core.utils.exceptions import CheckError


class _Checks:
    checks = []


class OperationsStage:
    """The operations stage of the checks."""

    def __init__(self, creation_stage):
        """Create an intance of the stage.

        Parameters
        ----------
        creation_stage: `CreationStage`
            An instance of the previous stage.

        """
        self.c_stage = creation_stage

    @staticmethod
    def checks():
        return _Checks.checks

    @register_check(_Checks.checks)
    def check_write(self):
        """Check if database supports write operation... """
        database = self.c_stage.instance

        try:
            database.write('test', {'index': 'value'})
        except Exception as ex:
            raise CheckError(str(ex))

        return "Success", ""

    @register_check(_Checks.checks)
    def check_read(self):
        """Check if database supports read operation... """
        database = self.c_stage.instance

        try:
            result = database.read('test', {'index': 'value'})
        except Exception as ex:
            raise CheckError(str(ex))

        value = result[0]['index']
        if value != 'value':
            raise CheckError("Expected 'value', received {}.".format(value))

        return "Success", ""

    @register_check(_Checks.checks)
    def check_count(self):
        """Check if database supports count operation... """
        database = self.c_stage.instance

        count = database.count('test', {'index': 'value'})

        if count != 1:
            raise CheckError("Expected 1 hit, received {}.".format(count))

        return "Success", ""

    @register_check(_Checks.checks)
    def check_remove(self):
        """Check if database supports delete operation... """
        database = self.c_stage.instance

        database.remove('test', {'index': 'value'})
        remaining = database.count('test', {'index': 'value'})

        if remaining:
            raise CheckError("{} items remaining.".format(remaining))

        return "Success", ""
