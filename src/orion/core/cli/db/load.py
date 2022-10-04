#!/usr/bin/env python
# pylint: disable=too-few-public-methods
"""
Storage import tool
===================

Import database content from a file.

"""
import argparse
import logging
import os
import pprint

from orion.core.io import experiment_builder
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.base import setup_storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DESCRIPTION = "Import storage"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    serve_parser = parser.add_parser("load", help=DESCRIPTION, description=DESCRIPTION)

    serve_parser.add_argument(
        "-c",
        "--config",
        type=argparse.FileType("r"),
        metavar="path-to-config",
        help="Orion config file, used to open database",
    )

    serve_parser.add_argument(
        "-e",
        "--exp",
        type=str,
        default=None,
        help="Experiment to import (default: all experiments are imported)",
    )

    serve_parser.add_argument(
        "file",
        type=str,
        help="File to import",
    )

    serve_parser.add_argument(
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

    serve_parser.set_defaults(func=main)

    return serve_parser


def main(args):
    """Script to import storage"""
    experiment = args["exp"]
    resolve = args["resolve"]
    storage = setup_storage(experiment_builder.get_cmd_config(args).get("storage"))
    dst_db = storage._db
    logger.info(f"Loaded dst {dst_db}")
    src_db = PickledDB(host=os.path.abspath(args["file"]))
    logger.info(f"Loaded src {src_db}")
    with src_db.locked_database(write=False) as src_database:
        src_collection_names = set(src_database._db.keys())
        if experiment is None:
            # TODO How to manage collisions
            for collection_name in src_database._db.keys():
                data = src_database.read(collection_name)
                logger.info(
                    f"Importing collection {collection_name} " f"({len(data)} entries)"
                )
                dst_db.write(collection_name, data)
        else:
            logger.info(f"Filter experiment {experiment}")
            # Get experiments with given name
            assert "experiments" in src_collection_names
            src_experiments = {
                d["_id"]: d
                for d in src_database.read("experiments")
                if d["name"] == experiment
            }
            if not src_experiments:
                logger.info(f"Experiment not found in src: {experiment}")
                return

            all_dst_experiments = dst_db.read("experiments")
            dst_experiments = {
                d["_id"]: d for d in all_dst_experiments if d["name"] == experiment
            }
            if resolve == "ignore" and dst_experiments:
                # Found experiment name in src and dst.
                # Ignore experiment.
                logger.info(f"Experiment already in dst, ignored: {experiment}")
                return

            logger.info(
                f"Found {len(src_experiments)} experiment(s) named {experiment}"
            )
            # Get data related to experiment
            src_related_data = {
                collection_name: [
                    element
                    for element in src_database.read(collection_name)
                    if element.get("experiment", None) in src_experiments
                ]
                for collection_name in sorted(src_collection_names - {"experiments"})
            }
            if resolve == "overwrite":
                if dst_experiments:
                    logger.info(
                        f"Overwrite {len(dst_experiments)} experiment(s) "
                        f"named {experiment} in dst"
                    )
                    # We must remove experiment data in dst
                    if isinstance(dst_db, MongoDB):
                        dst_coll_names = {
                            data["name"] for data in dst_db._db.list_collections()
                        }
                    elif isinstance(dst_db, PickledDB):
                        with dst_db.locked_database(write=False) as dst_database:
                            dst_coll_names = set(dst_database._db.keys())
                    else:
                        dst_coll_names = set(dst_db._db.keys())
                    dst_related_data = {
                        collection_name: [
                            element
                            for element in dst_db.read(collection_name)
                            if element.get("experiment", None) in dst_experiments
                        ]
                        for collection_name in (dst_coll_names - {"experiments"})
                    }
                    dst_related_data["experiments"] = list(dst_experiments.values())
                    for collection_name, collection_data in dst_related_data.items():
                        for data in collection_data:
                            dst_db.remove(collection_name, query={"_id": data["_id"]})
            else:
                logger.info(
                    f"Bump {len(src_experiments)} experiment(s) "
                    f"named {experiment} from src"
                )
                # resolve == "bump"
                # Bump version and update experiment indices
                next_id = (
                    max((data["_id"] for data in all_dst_experiments), default=0) + 1
                )
                old_to_new_id = {}
                for src_exp in src_experiments.values():
                    old_to_new_id[src_exp["_id"]] = next_id
                    next_id += 1
                    src_exp["version"] += 1
                    src_exp["_id"] = old_to_new_id[src_exp["_id"]]
                for src_col_data in src_related_data.values():
                    for element in src_col_data:
                        if "experiment" in element:
                            element["experiment"] = old_to_new_id[element["experiment"]]
            # Then we can insert experiment data from src into dst
            # Remove "_id" fields from data to insert
            for coll_data in src_related_data.values():
                for data in coll_data:
                    del data["_id"]
            # Add experiment to data to insert
            # Keep "_id" as it is already updated
            src_related_data["experiments"] = list(src_experiments.values())
            pprint.pprint(src_related_data)
            for collection_name, collection_data in src_related_data.items():
                logger.info(
                    f"Load {len(collection_data)} data "
                    f"from collection {collection_name} to dst"
                )
                dst_db.write(collection_name, collection_data)
