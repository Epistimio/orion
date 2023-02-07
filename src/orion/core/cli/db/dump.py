#!/usr/bin/env python
# pylint: disable=,protected-access
"""
Storage export tool
===================

Export database content into a file.

"""
import logging
import os

from orion.core.cli import base as cli
from orion.core.io import experiment_builder
from orion.core.io.database import DatabaseError
from orion.core.io.database.pickleddb import PickledDB
from orion.storage.base import setup_storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DESCRIPTION = "Export storage"

COLLECTIONS = {"experiments", "algo", "benchmarks", "trials"}


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    dump_parser = parser.add_parser("dump", help=DESCRIPTION, description=DESCRIPTION)

    cli.get_basic_args_group(dump_parser)

    dump_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dump.pkl",
        help="Output file path (default: dump.pkl)",
    )

    dump_parser.set_defaults(func=main)

    return dump_parser


def main(args):
    """Script to dump storage"""
    config = experiment_builder.get_cmd_config(args)
    storage = setup_storage(config.get("storage"))
    orig_db = storage._db
    logger.info(f"Loaded {orig_db}")
    dump_database(
        orig_db,
        args["output"],
        experiment=config.get("name"),
        version=config.get("version"),
    )


def dump_database(orig_db, dump_host, experiment=None, version=None):
    """Dump a database
    :param orig_db: database to dump
    :param dump_host: file path to dump into (dumped file will be a pickled file)
    :param experiment: (optional) name of experiment to dump (by default, full database is dumped)
    :param version: (optional) version of experiment to dump
    """
    dump_host = os.path.abspath(dump_host)
    if isinstance(orig_db, PickledDB) and dump_host == os.path.abspath(orig_db.host):
        raise DatabaseError(f"Source ({orig_db.host}) and target ({dump_host}) cannot be the same.")
    dst_storage = setup_storage({"database": {"host": dump_host, "type": "pickleddb"}})
    db = dst_storage._db
    logger.info(f"Dump to {db}")
    if isinstance(orig_db, PickledDB):
        with orig_db.locked_database(write=False) as database:
            _dump(database, db, COLLECTIONS, experiment, version)
    else:
        _dump(orig_db, db, COLLECTIONS, experiment, version)


def _dump(src_db, dst_db, collection_names, experiment=None, version=None):
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
        query = {"name": experiment}
        if version is not None:
            query["version"] = version
        experiments = src_db.read("experiments", query)
        if not experiments:
            raise DatabaseError(
                f"No experiment found with query {query}. Nothing to dump."
            )
        if len(experiments) > 1:
            exp_data = sorted(experiments, key=lambda d: d["version"])[0]
        else:
            (exp_data,) = experiments
        logger.info(f"Found experiment {exp_data['name']}.{exp_data['version']}")
        # Dump selected experiments
        logger.info(f"Dumping experiment {experiment}")
        dst_db.write("experiments", exp_data)
        # Dump data related to selected experiments (do not dump other experiments)
        for collection_name in sorted(collection_names - {"experiments"}):
            filtered_data = [
                element
                for element in src_db.read(collection_name)
                if element.get("experiment", None) == exp_data["_id"]
            ]
            dst_db.write(collection_name, filtered_data)
            logger.info(
                f"Written {len(filtered_data)} filtered data "
                f"for collection {collection_name}"
            )
