#!/usr/bin/env python
# pylint: disable=,protected-access
"""
Storage export tool
===================

Export database content into a file.

"""
import logging

from orion.core.cli import base as cli
from orion.core.io import experiment_builder
from orion.core.worker.storage_backup import dump_database
from orion.storage.base import setup_storage

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DESCRIPTION = "Export storage"


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
    logger.info(f"Loaded {storage._db}")
    dump_database(
        storage,
        args["output"],
        name=config.get("name"),
        version=config.get("version"),
    )
