#!/usr/bin/env python
"""
Creation stage for database checks
==================================

Checks for the creation of a `Database` object.

"""

from orion.core.io.database import database_factory
from orion.core.utils.exceptions import CheckError


class CreationStage:
    """The creation stage of the checks."""

    def __init__(self, presence_stage):
        """Create an instance of the stage.

        Parameters
        ----------
        presence_stage: `PresenceStage`
            An instance of the previous stage.

        """
        self.p_stage = presence_stage
        self.instance = None

    def checks(self):
        """Return checks."""
        yield self.check_database_creation

    def check_database_creation(self):
        """Check if database of specified type can be created."""
        database = self.p_stage.db_config
        db_type = database.pop("type")

        try:
            db = database_factory.create(db_type, **database)
        except ValueError as ex:
            raise CheckError(str(ex)) from ex

        if not db.is_connected:
            raise CheckError("Database failed to connect after creation.")

        self.instance = db

        return "Success", ""

    def post_stage(self):
        """Print the created database."""
        print("DB instance", self.instance)
