#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.checks.presence` -- Presence stage for database checks
===========================================================================

.. module:: presence
    :platform: Unix
    :synopsis: Checks for the creation of a `Database` object.

"""

from orion.core.utils.decorators import register_check
from orion.core.utils.exceptions import CheckError


class _Checks:
    checks = []


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
        self.db_config = None

    @staticmethod
    def checks():
        """Return the registered checks."""
        return _Checks.checks

    @register_check(_Checks.checks, "check if oríon's default options are present... ")
    def check_default_config(self):
        """Check if database supports delete operation."""
        config = self.builder.fetch_default_options()

        if 'database' not in config:
            return "Skipping", "No default configuration found for database."

        self.db_config = config['database']

        return "Success", ""

    @register_check(_Checks.checks, "check if oríon's environment variables have been set... ")
    def check_environment_vars(self):
        """Check if database supports delete operation."""
        config = self.builder.fetch_env_vars()

        config = config['database']
        names = ['type', 'name', 'host']

        for name in names:
            if name not in config or config[name] is None:
                return "Skipping", "Missing {} environment variable.".format(name)

        self.db_config = config

        return "Success", ""

    @register_check(_Checks.checks, "Check if configuration file has"
                                    " valid database configuration... """)
    def check_configuration_file(self):
        """Check if configuration file has valid database configuration."""
        config = self.builder.fetch_file_config(self.cmdargs)

        try:
            if not len(config):
                raise CheckError("Missing configuration file.")

            if 'database' not in config:
                raise CheckError("No database found in configuration file.")

            config = config['database']
            names = ['type', 'name', 'host']

            for name in names:
                if name not in config or config[name] is None:
                    raise CheckError("Missing {} inside configuration.".format(name))
        except CheckError as ex:
            if self.db_config is not None:
                return "Skipping", "No configuration file found, using previous."
            else:
                raise ex

        self.db_config = config
        return "Success", ""

    def post_stage(self):
        """Print the current config."""
        print("Using configuration: {}".format(self.db_config))
