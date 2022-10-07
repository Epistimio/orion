#!/usr/bin/env python
# pylint: disable=too-few-public-methods
"""
Storage export tool
===================

Export database content into a file.

"""
import argparse
import logging
import os

from orion.core.io import experiment_builder
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.base import setup_storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DESCRIPTION = "Export storage"


def dump_database(orig_db, dump_host, experiment=None):
    dump_host = os.path.abspath(dump_host)
    if isinstance(orig_db, PickledDB) and dump_host == os.path.abspath(orig_db.host):
        logger.info("Cannot dump pickleddb to itself.")
        return 0
    dst_storage = setup_storage({"database": {"host": dump_host, "type": "pickleddb"}})
    db = dst_storage._db
    logger.info(f"Dump to {db}")
    if isinstance(orig_db, PickledDB):
        with orig_db.locked_database(write=False) as database:
            collection_names = set(database._db.keys())
            _dump(database, db, collection_names, experiment)
    else:
        if isinstance(orig_db, MongoDB):
            collection_names = (data["name"] for data in orig_db._db.list_collections())
        else:
            collection_names = orig_db._db.keys()
        _dump(orig_db, db, set(collection_names), experiment)


def _dump(src_db, dst_db, collection_names, experiment=None):
    """
    Dump data from database to db.
    :param src_db: input database
    :param dst_db: output database
    :param collection_names: set of collection names to dump
    :param experiment: (optional) if provided, dump only
        data related to experiment with this name
    """
    # Get collection names in a set
    if experiment is None:
        # Nothing to filter, dump everything
        for collection_name in collection_names:
            logger.info(f"Dumping collection {collection_name}")
            data = src_db.read(collection_name)
            dst_db.write(collection_name, data)
    else:
        # Get experiments with given name
        assert "experiments" in collection_names
        experiments = src_db.read("experiments", {"name": experiment})
        logger.info(f"Found {len(experiments)} experiment(s) named {experiment}")
        # Dump selected experiments
        logger.info(f"Dumping experiment {experiment}")
        dst_db.write("experiments", experiments)
        # Do not dump other experiments
        collection_names.remove("experiments")
        # Dump data related to selected experiments
        exp_indices = {exp["_id"] for exp in experiments}
        for collection_name in sorted(collection_names):
            logger.info(f"Dumping collection {collection_name}")
            filtered_data = [
                element
                for element in src_db.read(collection_name)
                if element.get("experiment", None) in exp_indices
            ]
            dst_db.write(collection_name, filtered_data)
            logger.info(
                f"Written {len(filtered_data)} filtered data "
                f"for collection {collection_name}"
            )


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    serve_parser = parser.add_parser("dump", help=DESCRIPTION, description=DESCRIPTION)

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
        help="Experiment to dump (default: all experiments are exported)",
    )

    serve_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dump.pkl",
        help="Output file path (default: dump.pkl)",
    )

    serve_parser.set_defaults(func=main)

    return serve_parser


def main(args):
    """Script to dump storage"""
    storage = setup_storage(experiment_builder.get_cmd_config(args).get("storage"))
    orig_db = storage._db
    logger.info(f"Loaded {orig_db}")
    dump_database(orig_db, args["output"], experiment=args["exp"])
