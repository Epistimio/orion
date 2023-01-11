#!/usr/bin/env python
"""
Operations stage for database checks
====================================

Checks for the operations of a `Database` object.

"""

from orion.core.utils.exceptions import CheckError


class OperationsStage:
    """The operations stage of the checks."""

    def __init__(self, creation_stage):
        """Create an instance of the stage.

        Parameters
        ----------
        creation_stage: `CreationStage`
            An instance of the previous stage.

        """
        self.c_stage = creation_stage

    def checks(self):
        """Return checks."""
        yield self.check_write
        yield self.check_read
        yield self.check_count
        yield self.check_remove

    def check_write(self):
        """Check if database supports write operation."""
        database = self.c_stage.instance

        try:
            database.write("test", {"index": "value"})
        except Exception as ex:
            raise CheckError(str(ex)) from ex

        return "Success", ""

    def check_read(self):
        """Check if database supports read operation."""
        database = self.c_stage.instance

        try:
            result = database.read("test", {"index": "value"})
        except Exception as ex:
            raise CheckError(str(ex)) from ex

        if len(result) == 0:
            raise CheckError("Expected 'value', received nothing.")

        return "Success", ""

    def check_count(self):
        """Check if database supports count operation."""
        database = self.c_stage.instance

        count = database.count("test", {"index": "value"})

        if count != 1:
            raise CheckError(f"Expected 1 hit, received {count}.")

        return "Success", ""

    def check_remove(self):
        """Check if database supports delete operation."""
        database = self.c_stage.instance

        database.remove("test", {"index": "value"})
        remaining = database.count("test", {"index": "value"})

        if remaining:
            raise CheckError(f"{remaining} items remaining.")

        return "Success", ""

    def post_stage(self):
        """Do nothing."""
