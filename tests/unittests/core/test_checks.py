#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.cli.checks`."""
from orion.core.cli.checks.presence import PresenceStage
from orion.core.cli.checks.operations import OperationsStage
from orion.core.cli.checks.creation import CreationStage
from orion.core.io.experiment_builder import ExperimentBuilder

import pytest


@pytest.fixture
def presence_stage(yaml_config):
    """Return a PresenceStage instance."""
    exp_builder = ExperimentBuilder()
    return PresenceStage(exp_builder, yaml_config)


@pytest.fixture
def creation_stage(presence_stage):

