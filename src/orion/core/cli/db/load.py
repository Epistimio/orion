#!/usr/bin/env python
"""
Storage import tool
===================

Import database content from a file.

"""
import logging

from orion.core.cli import base as cli
from orion.core.io import experiment_builder
from orion.storage.backup import load_database
from orion.storage.base import setup_storage

logger = logging.getLogger(__name__)

DESCRIPTION = "Import storage"


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
        help="Strategy to resolve conflicts: "
        "'ignore', 'overwrite' or 'bump' "
        "(bump version of imported experiment). "
        "When overwriting, prior trials will be deleted. "
        "If not specified, an exception will be raised on any conflict detected.",
    )

    load_parser.set_defaults(func=main)

    return load_parser


def main(args):
    """Script to import storage"""
    config = experiment_builder.get_cmd_config(args)
    storage = setup_storage(config.get("storage"))
    logger.info(f"Loaded dst {storage}")
    load_database(
        storage,
        load_host=args["file"],
        resolve=args["resolve"],
        name=config.get("name"),
        version=config.get("version"),
    )
