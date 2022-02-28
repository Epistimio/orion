#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backward compatibility utils
============================

Helper functions to support backward compatibility.

"""
from __future__ import annotations

import copy
import logging
import pprint
import typing
from typing import Any, Sequence, Type

import orion.core
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser
from orion.core.worker.trial import Trial

if typing.TYPE_CHECKING:
    from orion.algo.base import BaseAlgorithm
    from orion.client.experiment import ExperimentClient
    from orion.core.io.database import Database

log = logging.getLogger(__name__)


def update_user_args(metadata: dict[str, Any]) -> None:
    """Make sure user script is not removed from metadata"""
    if (
        "user_script" in metadata
        and metadata["user_script"] not in metadata["user_args"]
    ):
        log.debug("Updating user_args for backward compatibility")
        metadata["user_args"] = [metadata["user_script"]] + metadata["user_args"]
        log.debug(pprint.pformat(metadata["user_args"]))


def populate_priors(metadata: dict[str, Any]) -> None:
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


def update_max_broken(config: dict[str, Any]) -> None:
    """Set default max_broken if None (in v <= v0.1.9)"""
    if not config.get("max_broken", None):
        config["max_broken"] = orion.core.config.experiment.max_broken
        log.debug(
            "Updating max_broken for backward compatibility: %s", config["max_broken"]
        )


def populate_space(config: dict[str, Any], force_update: bool = True) -> None:
    """Add the space definition at the root of config."""
    if "space" in config and not force_update:
        return

    populate_priors(config["metadata"])
    # Overwrite space to make sure to include changes from user_args
    if "priors" in config["metadata"]:
        log.debug("Updating space for backward compatibility")
        log.debug(pprint.pformat(config["metadata"]["priors"]))
        config["space"] = config["metadata"]["priors"]


def db_is_outdated(database: Database) -> bool:
    """Return True if the database scheme is outdated."""
    deprecated_indices = [
        ("name", "metadata.user"),
        ("name", "metadata.user", "version"),
        "name_1_metadata.user_1",
        "name_1_metadata.user_1_version_1",
    ]

    index_information = database.index_information("experiments")
    return any(index in deprecated_indices for index in index_information.keys())


def update_db_config(config: dict[str, Any]) -> None:
    """Merge DB config back into storage config"""
    config.setdefault("storage", orion.core.config.storage.to_dict())
    if "database" in config:
        log.debug("Updating db config for backward compatibility")
        config["storage"] = {"type": "legacy"}
        config["storage"]["database"] = config.pop("database")
        log.debug(pprint.pformat(config["storage"]))


def get_algo_requirements(algorithm: type[BaseAlgorithm]) -> dict[str, str | None]:
    """Return a dict() of requirements of the algorithm based on interface < v0.1.10"""
    if not hasattr(algorithm, "requires"):
        return dict(
            type_requirement=algorithm.requires_type,
            shape_requirement=algorithm.requires_shape,
            dist_requirement=algorithm.requires_dist,
        )

    log.warning(
        "Algorithm.requires is deprecated and will stop being supporting in v0.3."
    )
    requirements = algorithm.requires  # type: ignore
    requirements = requirements if isinstance(requirements, list) else [requirements]

    log.debug("Algorithm requirements: %s", requirements)

    requirements = copy.deepcopy(requirements)

    if "linear" in requirements:
        dist_requirement: str | None = "linear"
        del requirements[requirements.index("linear")]
    else:
        dist_requirement = None

    if "flattened" in requirements:
        shape_requirement: str | None = "flattened"
        del requirements[requirements.index("flattened")]
    else:
        shape_requirement = None

    if requirements:
        assert len(requirements) == 1
        type_requirement: str | None = requirements[0]
    else:
        type_requirement = None

    requirements = dict(
        type_requirement=type_requirement,
        shape_requirement=shape_requirement,
        dist_requirement=dist_requirement,
    )

    log.debug("Algorithm requirements in new format:\n%s", pprint.pformat(requirements))

    return requirements


def port_algo_config(config: str | dict[str, Any]) -> dict[str, Any]:
    """Convert algorithm configuration to be compliant with factory interface

    Examples
    --------
    >>> port_algo_config('algo_name')
    {'of_type': 'algo_name'}
    >>> port_algo_config({'algo_name': {'some': 'args'}})
    {'of_type': 'algo_name', 'some': 'args'}
    >>> port_algo_config({'of_type': 'algo_name', 'some': 'args'})
    {'of_type': 'algo_name', 'some': 'args'}

    """
    config = copy.deepcopy(config)
    new_config: dict[str, Any]
    if isinstance(config, dict) and len(config) == 1:
        algo_name, algo_config = next(iter(config.items()))
        assert isinstance(algo_config, dict)
        new_config = algo_config
        new_config["of_type"] = algo_name
    elif isinstance(config, str):
        new_config = {"of_type": config}
    else:
        new_config = config
    return new_config


def algo_observe(algo: BaseAlgorithm, trials: Sequence[Trial], results: Sequence[dict]):
    """Convert trials so that algo can observe with legacy format (trials, results)."""
    for trial, trial_results in zip(trials, results):
        for name, trial_result in trial_results.items():
            if trial.exp_working_dir is None:
                trial.exp_working_dir = "/nothing"
            trial.status = "completed"
            trial.results.append(Trial.Result(name=name, type=name, value=trial_result))

    algo.observe(trials)


def ensure_trial_working_dir(experiment: ExperimentClient, trial: Trial) -> None:
    """If the trial's exp working dir is not set, set it to current experiment's working dir."""
    if not trial.exp_working_dir:
        trial.exp_working_dir = experiment.working_dir
