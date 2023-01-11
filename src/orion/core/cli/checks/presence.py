#!/usr/bin/env python
"""
Presence stage for database checks
==================================

Checks for the presence of a configuration.

"""

import orion.core
from orion.core.io import experiment_builder
from orion.core.utils import backward


class PresenceStage:
    """The presence stage of the checks."""

    def __init__(self, cmdargs):
        """Create an instance of the stage."""
        self.cmdargs = cmdargs
        self.db_config = {}

    def checks(self):
        """Return the registered checks."""
        yield self.check_default_config
        yield self.check_configuration_file

    def check_default_config(self):
        """Check for a configuration inside the default paths."""
        config = orion.core.config.to_dict()

        backward.update_db_config(config)

        if "database" not in config.get("storage", {}):
            return "Skipping", "No default configuration found for database."

        self.db_config = config["storage"]["database"]
        print("\n   ", self.db_config)

        return "Success", ""

    def check_configuration_file(self):
        """Check if configuration file has valid database configuration."""
        config = experiment_builder.get_cmd_config(self.cmdargs)

        if len(config) == 0:
            return "Skipping", "Missing configuration file."

        if "database" not in config.get("storage", {}):
            return "Skipping", "No database found in configuration file."

        config = config["storage"]["database"]
        names = ["type", "name", "host", "port"]

        if not any(name in config for name in names):
            return "Skipping", "No configuration value found inside `database`."

        self.db_config.update(config)

        print("\n   ", config)

        return "Success", ""

    def post_stage(self):
        """Print the current config."""
        print(f"Using configuration: {self.db_config}")
