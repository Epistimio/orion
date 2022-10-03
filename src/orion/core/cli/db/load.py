#!/usr/bin/env python
# pylint: disable=too-few-public-methods
"""
Storage import tool
===================

Import database content from a file.

"""
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DESCRIPTION = "Import storage"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    serve_parser = parser.add_parser("load", help=DESCRIPTION, description=DESCRIPTION)

    serve_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
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
    """Starts an application server to serve http requests"""
    print("Current args")
    print(args)
