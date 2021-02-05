#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backward compatibility utils
============================

Helper functions to support backward compatibility.

"""
import copy
import logging
import pprint

import orion.core
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser

log = logging.getLogger(__name__)


def update_user_args(metadata):
    """Make sure user script is not removed from metadata"""
    if (
        "user_script" in metadata
        and metadata["user_script"] not in metadata["user_args"]
    ):
        log.debug("Updating user_args for backward compatibility")
        metadata["user_args"] = [metadata["user_script"]] + metadata["user_args"]
        log.debug(pprint.pformat(metadata["user_args"]))


def populate_priors(metadata):
    """Compute parser state and priors based on user_args and populate metadata."""
    if "user_args" not in metadata:
        return

    update_user_args(metadata)

    parser = OrionCmdlineParser(
        orion.core.config.worker.user_script_config, allow_non_existing_files=True
    )
    if "parser" in metadata:
        # To keep configs like config user_script_config
        parser.config_prefix = metadata["parser"]["config_prefix"]
    parser.parse(metadata["user_args"])

    log.debug("Updating parser for backward compatibility")
    metadata["parser"] = parser.get_state_dict()
    log.debug(pprint.pformat(metadata["parser"]))

    log.debug("Updating priors for backward compatibility")
    metadata["priors"] = dict(parser.priors)
    log.debug(pprint.pformat(metadata["priors"]))


def update_max_broken(config):
    """Set default max_broken if None (in v <= v0.1.9)"""
    if not config.get("max_broken", None):
        config["max_broken"] = orion.core.config.experiment.max_broken
        log.debug(
            "Updating max_broken for backward compatibility: %s", config["max_broken"]
        )


def populate_space(config, force_update=True):
    """Add the space definition at the root of config."""
    if "space" in config and not force_update:
        return

    populate_priors(config["metadata"])
    # Overwrite space to make sure to include changes from user_args
    if "priors" in config["metadata"]:
        log.debug("Updating space for backward compatibility")
        log.debug(pprint.pformat(config["metadata"]["priors"]))
        config["space"] = config["metadata"]["priors"]


def db_is_outdated(database):
    """Return True if the database scheme is outdated."""
    deprecated_indices = [
        ("name", "metadata.user"),
        ("name", "metadata.user", "version"),
        "name_1_metadata.user_1",
        "name_1_metadata.user_1_version_1",
    ]

    index_information = database.index_information("experiments")
    return any(index in deprecated_indices for index in index_information.keys())


def update_db_config(config):
    """Merge DB config back into storage config"""
    config.setdefault("storage", orion.core.config.storage.to_dict())
    if "database" in config:
        log.debug("Updating db config for backward compatibility")
        config["storage"] = {"type": "legacy"}
        config["storage"]["database"] = config.pop("database")
        log.debug(pprint.pformat(config["storage"]))


def get_algo_requirements(algorithm):
    """Return a dict() of requirements of the algorithm based on interface < v0.1.10"""
    if hasattr(algorithm, "requires"):
        log.warning(
            "Algorithm.requires is deprecated and will stop being supporting in v0.3."
        )
        requirements = algorithm.requires
        requirements = (
            requirements if isinstance(requirements, list) else [requirements]
        )

        log.debug("Algorithm requirements: %s", requirements)

        requirements = copy.deepcopy(requirements)

        if "linear" in requirements:
            dist_requirement = "linear"
            del requirements[requirements.index("linear")]
        else:
            dist_requirement = None

        if "flattened" in requirements:
            shape_requirement = "flattened"
            del requirements[requirements.index("flattened")]
        else:
            shape_requirement = None

        if requirements:
            assert len(requirements) == 1
            type_requirement = requirements[0]
        else:
            type_requirement = None

        requirements = dict(
            type_requirement=type_requirement,
            shape_requirement=shape_requirement,
            dist_requirement=dist_requirement,
        )

        log.debug(
            "Algorithm requirements in new format:\n%s", pprint.pformat(requirements)
        )

        return requirements

    return dict(
        type_requirement=algorithm.requires_type,
        shape_requirement=algorithm.requires_shape,
        dist_requirement=algorithm.requires_dist,
    )
