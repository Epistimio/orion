#!/usr/bin/env python
# pylint: disable=too-few-public-methods
"""
Storage export tool
===================

Export database content into a file.

"""
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DESCRIPTION = "Export storage"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    serve_parser = parser.add_parser("dump", help=DESCRIPTION, description=DESCRIPTION)

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
        help="Experiment to dump (default: all experiments are exported)",
    )

    serve_parser.set_defaults(func=main)

    return serve_parser


def main(args):
    """Starts an application server to serve http requests"""
    print("Current args")
    print(args)
