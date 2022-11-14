#!/usr/bin/env python
"""
Module to upgrade DB schemes
============================

Upgrade the scheme of the databases

"""
import argparse
import logging
import sys

from orion.core.io import experiment_builder
from orion.core.io.config import ConfigurationError
from orion.core.io.database.ephemeraldb import EphemeralCollection
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.legacy import Legacy

log = logging.getLogger(__name__)
SHORT_DESCRIPTION = "Upgrade the database scheme"


# TODO: Move somewhere else to share with `db setup`.
def ask_question(question, default=None):
    """Ask a question to the user and receive an answer.

    Parameters
    ----------
    question: str
        The question to be asked.
    default: str
        The default value to use if the user enters nothing.

    Returns
    -------
    str
        The answer provided by the user.

    """
    if default is not None:
        question = f"{question} (default: {default}) "

    answer = input(question)

    if answer.strip() == "":
        return default

    return answer


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    upgrade_db_parser = parser.add_parser(
        "upgrade", help=SHORT_DESCRIPTION, description=SHORT_DESCRIPTION
    )

    upgrade_db_parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="user provided " "orion configuration file",
    )

    upgrade_db_parser.add_argument(
        "-f", "--force", action="store_true", help="Don't prompt user"
    )

    upgrade_db_parser.set_defaults(func=main)

    return upgrade_db_parser


def main(args):
    """Upgrade the databases for current version"""
    print(
        "Upgrading your database may damage your data. Make sure to make a backup before the "
        "upgrade and stop any other process that may read/write the database during the upgrade."
    )

    if not args.get("force"):
        action = ""
        while action not in ["y", "yes", "no", "n"]:
            action = ask_question("Do you wish to proceed? (y/N)", "N").lower()

        if action in ["no", "n"]:
            sys.exit(0)

    config = experiment_builder.get_cmd_config(args)
    storage_config = config.get("storage")

    if storage_config is None:
        storage_config = {"type": "legacy"}

    storage_config["setup"] = False

    builder = experiment_builder.ExperimentBuilder(storage_config)

    upgrade_db_specifics(builder.storage)

    print("Updating documents...")
    upgrade_documents(builder.storage)
    print("Database upgrade completed successfully")


def upgrade_db_specifics(storage):
    """Make upgrades that are specific to some backends"""
    if isinstance(storage, Legacy):
        database = storage._db  # pylint: disable=protected-access
        print("Updating indexes...")
        update_indexes(database)
        if isinstance(database, PickledDB):
            print("Updating pickledb scheme...")
            upgrade_pickledb(database)
        elif isinstance(database, MongoDB):
            print("Updating mongodb scheme...")
            upgrade_mongodb(database)


def upgrade_documents(storage):
    """Upgrade scheme of the documents"""
    for experiment in storage.fetch_experiments({}):
        add_version(experiment)
        uid = experiment.pop("_id")
        algorithm = None
        if "algorithms" in experiment:
            algorithm = experiment.pop("algorithms")
        if "algorithm" in experiment:
            algorithm = experiment.pop("algorithm")

        if algorithm is None:
            raise ConfigurationError(
                "The data was corrupted, as there was no algorithm in the experiment"
            )

        storage.update_experiment(uid=experiment, **experiment)
        storage.initialize_algorithm_lock(uid, algorithm)

        for trial in storage.fetch_trials(uid=uid):
            # trial_config = trial.to_dict()
            assert trial.id_override is not None
            # This will overwrite the trial with correct dict structure
            # (ex: have _id set to id_override and id set to trial.hash_name == trial.id.
            # storage.update_trial(trial, uid=trial.id_override)
            # pylint: disable=protected-access
            storage._db.write(
                "trials", data=trial.to_dict(), query={"_id": trial.id_override}
            )


def add_version(experiment):
    """Add version 1 if not present"""
    experiment.setdefault("version", 1)


def update_indexes(database):
    """Remove user from unique indices.

    This is required for migration to v0.1.6+
    """
    # For backward compatibility
    index_info = database.index_information("experiments")
    deprecated_indices = [
        ("name", "metadata.user"),
        ("name", "metadata.user", "version"),
        "name_1_metadata.user_1",
        "name_1_metadata.user_1_version_1",
    ]

    for deprecated_idx in deprecated_indices:
        if deprecated_idx in index_info:
            database.drop_index("experiments", deprecated_idx)


# pylint: disable=unused-argument
def upgrade_mongodb(database):
    """Update mongo specific db scheme."""


def upgrade_pickledb(database):
    """Update pickledb specific db scheme."""
    # pylint: disable=protected-access
    def upgrade_state(self, state):
        """Set state while ensuring backward compatibility"""
        self._documents = state["_documents"]

        # if indexes are from <=v0.1.6
        if state["_indexes"] and isinstance(
            next(iter(state["_indexes"].keys())), tuple
        ):
            self._indexes = {}
            for keys, values in state["_indexes"].items():
                if isinstance(keys, str):
                    self._indexes[keys] = values
                # Convert keys that were registered with old index signature
                else:
                    keys = [(key, None) for key in keys]
                    self.create_index(keys, unique=True)
        else:
            self._indexes = state["_indexes"]

    old_setstate = getattr(EphemeralCollection, "__setstate__", None)
    EphemeralCollection.__setstate__ = upgrade_state

    document = database.read("experiments", {})[0]
    # One document update is enough to fix all collections
    database.write("experiments", document, query={"_id": document["_id"]})

    if old_setstate is not None:
        EphemeralCollection.__setstate__ = old_setstate
    else:
        del EphemeralCollection.__setstate__
