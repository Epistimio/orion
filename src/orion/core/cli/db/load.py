#!/usr/bin/env python
# pylint: disable=too-few-public-methods
"""
Storage import tool
===================

Import database content from a file.

"""
import logging
import os

from orion.core.cli import base as cli
from orion.core.io import experiment_builder
from orion.core.worker.trial import Trial
from orion.storage.base import setup_storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DESCRIPTION = "Import storage"

COLLECTIONS = {"experiments", "algo", "benchmarks", "trials"}
EXPERIMENT_RELATED_COLLECTIONS = {"algo", "trials"}


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    load_parser = parser.add_parser("load", help=DESCRIPTION, description=DESCRIPTION)

    cli.get_basic_args_group(load_parser)

    load_parser.add_argument(
        "file",
        type=str,
        help="File to import",
    )

    load_parser.add_argument(
        "-r",
        "--resolve",
        type=str,
        choices=("ignore", "overwrite", "bump"),
        required=True,
        help="Strategy to resolve conflicts: "
        "'ignore', 'overwrite' or 'bump' "
        "(bump version of imported experiment). "
        "When overwriting, prior trials will be deleted.",
    )

    load_parser.set_defaults(func=main)

    return load_parser


def main(args):
    """Script to import storage"""
    config = experiment_builder.get_cmd_config(args)
    name = config.get("name")
    version = config.get("version")
    resolve = args["resolve"]

    dst_storage = setup_storage(config.get("storage"))
    dst_db = dst_storage._db
    logger.info(f"Loaded dst {dst_db}")

    src_storage = setup_storage(
        {"database": {"host": os.path.abspath(args["file"]), "type": "pickleddb"}}
    )
    src_db = src_storage._db
    logger.info(f"Loaded src {src_db}")

    with src_db.locked_database(write=False) as src_database:
        if name is None:
            # Import benchmarks
            load_benchmarks(src_database, dst_db, resolve)
            # Retrieve all src experiments for export
            experiments = src_database.read("experiments")
        else:
            # Find experiments based on given name and version
            query = {"name": name}
            if version is not None:
                query["version"] = version
            experiments = src_database.read("experiments", query)
            logger.info(
                f"Found {len(experiments)} experiment(s) in src with query: {query}"
            )
        # Import experiments
        for experiment in experiments:
            load_experiment(src_database, dst_db, experiment, resolve)
        logger.info(f"Imported collection experiments ({len(experiments)} entries)")


def load_benchmarks(src_database, dst_db, resolve):
    collection_name = "benchmarks"
    benchmarks = src_database.read(collection_name)
    for src_benchmark in benchmarks:
        load_benchmark(dst_db, src_benchmark, resolve)
    logger.info(f"Imported collection {collection_name} ({len(benchmarks)} entries)")


def load_benchmark(dst_db, src_benchmark, resolve):
    collection_name = "benchmarks"
    dst_benchmarks = dst_db.read(collection_name, {"name": src_benchmark["name"]})
    if dst_benchmarks:
        assert len(dst_benchmarks) == 1
        (dst_benchmark,) = dst_benchmarks
        if resolve == "ignore":
            logger.info(f'Ignored benchmark already in dst: {src_benchmark["name"]}')
            return
        elif resolve == "overwrite":
            logger.info(f'Overwrite benchmark in dst, name: {src_benchmark["name"]}')
            dst_db.remove(collection_name, {"_id": dst_benchmark["_id"]})
        elif resolve == "bump":
            return
            raise RuntimeError(
                "Can't bump benchmark version, as benchmarks do not currently support versioning."
            )
    # Delete benchmark database ID so that a new one will be generated on insertion
    del src_benchmark["_id"]
    # Insert benchmark
    dst_db.write(collection_name, src_benchmark)


def load_experiment(src_database, dst_db, experiment, resolve):
    dst_experiments = dst_db.read(
        "experiments", {"name": experiment["name"], "version": experiment["version"]}
    )
    if dst_experiments:
        assert len(dst_experiments) == 1
        (dst_experiment,) = dst_experiments
        if resolve == "ignore":
            logger.info(
                f'Ignored experiment already in dst: {experiment["name"]}.{experiment["version"]}'
            )
            return
        elif resolve == "overwrite":
            # We must remove experiment data in dst
            logger.info(
                f'Overwrite experiment in dst: {experiment["name"]}.{experiment["version"]}'
            )
            for collection in EXPERIMENT_RELATED_COLLECTIONS:
                dst_db.remove(collection, {"experiment": dst_experiment["_id"]})
            dst_db.remove("experiments", {"_id": dst_experiment["_id"]})
        elif resolve == "bump":
            old_version = experiment["version"]
            new_version = (
                max(
                    (
                        data["version"]
                        for data in dst_db.read(
                            "experiments", {"name": experiment["name"]}
                        )
                    ),
                    default=0,
                )
                + 1
            )
            experiment["version"] = new_version
            logger.info(
                f'Bumped version of src experiment: {experiment["name"]}, from {old_version} to {new_version}'
            )
    else:
        logger.info(f'Import experiment {experiment["name"]}.{experiment["version"]}')

    # Get data related to experiment to import.
    algos = src_database.read("algo", {"experiment": experiment["_id"]})
    trials = [
        Trial(**data)
        for data in src_database.read("trials", {"experiment": experiment["_id"]})
    ]

    # Generate new experiment ID
    next_experiment_id = (
        max((data["_id"] for data in dst_db.read("experiments")), default=0) + 1
    )
    experiment["_id"] = next_experiment_id
    # Update algos and trials: set new experiment ID and remove algo/trials database IDs,
    # so that new IDs will be generated at insertion.
    # Trial parents are identified using trial identifier (trial.id)
    # which is not related to trial database ID (trial.id_override).
    # So, we can safely remove trial database ID.
    for algo in algos:
        algo["experiment"] = next_experiment_id
        del algo["_id"]
    for trial in trials:
        trial.experiment = next_experiment_id
        trial.id_override = None

    # Write data
    dst_db.write("experiments", experiment)
    dst_db.write("algo", algos)
    dst_db.write("trials", [trial.to_dict() for trial in trials])
