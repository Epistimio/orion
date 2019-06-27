#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.checks.presence` -- Presence stage for database checks
===========================================================================

.. module:: presence
    :platform: Unix
    :synopsis: Checks for the presence of a configuration.

"""


class PresenceStage:
    """The presence stage of the checks."""

    def __init__(self, experiment_builder, cmdargs):
        """Create an instance of the stage.

        Parameters
        ----------
        experiment_builder: `ExperimentBuilder`
            An instance of `ExperimentBuilder` to fetch configs.

        """
        self.builder = experiment_builder
        self.cmdargs = cmdargs
        self.db_config = {}

    def checks(self):
        """Return the registered checks."""
        yield self.check_default_config
        yield self.check_environment_vars
        yield self.check_configuration_file

    def check_default_config(self):
        """Check for a configuration inside the default paths."""
        config = self.builder.fetch_default_options()

        if 'database' not in config:
            return "Skipping", "No default configuration found for database."

        self.db_config = config['database']
        print(self.db_config)

        return "Success", ""

    def check_environment_vars(self):
        """Check for a configuration inside the environment variables."""
        config = self.builder.fetch_env_vars()

        config = config['database']
        names = ['type', 'name', 'host']

        if not any(name in config for name in names):
            return "Skipping", "No environment variables found."

        self.db_config.update(config)
        print(self.db_config)

        return "Success", ""

    def check_configuration_file(self):
        """Check if configuration file has valid database configuration."""
        config = self.builder.fetch_file_config(self.cmdargs)

        if not len(config):
            return "Skipping", "Missing configuration file."

        if 'database' not in config:
            return "Skipping", "No database found in configuration file."

        config = config['database']
        names = ['type', 'name', 'host']

        if not any(name in config for name in names):
            return "Skipping", "No configuration value found inside `database`."

        self.db_config.update(config)

        return "Success", ""

    def post_stage(self):
        """Print the current config."""
        print("Using configuration: {}".format(self.db_config))
