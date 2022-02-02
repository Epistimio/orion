#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""User script that exits with code given in argument."""
import argparse
import sys
from xmlrpc.client import boolean


def execute():
    """Execute a simple pipeline as an example."""
    # 1. Get the wanted exit code
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exitcode", "-e", type=int, default=0, help="wanted exit code"
    )
    parser.add_argument(
        "--crash", "-c", action="store_true", help="make a crash happen"
    )
    parser.add_argument("-x", type=float, required=True)
    inputs = parser.parse_args()

    # 2. if needed, crash !
    if inputs.crash:
        print("%b" % False)  # black_box_fail.py : this line should crash

    # 3. exit with the wanted code
    sys.exit(inputs.exitcode)


if __name__ == "__main__":
    execute()
