#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`orion.core.cli.serve` -- Web application endpoint
=======================================================

.. module:: serve
   :platform: Unix
   :synopsis: Starts an http endpoint to serve requests

"""
import logging

log = logging.getLogger(__name__)

DESCRIPTION = "Starts HTTP endpoints"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    serve_parser = parser.add_parser('serve', help='serve help', description=DESCRIPTION)
    serve_parser.set_defaults(func=main)

    return serve_parser


def main(args):
    """Starts an application server to serve http requests"""
    try:
        pass
    except Exception as exception:
        log.error("An unexpected error happened", exception)
