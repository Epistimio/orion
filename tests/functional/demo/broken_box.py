#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple one dimensional example for a possible user's script."""
import argparse

from orion.client import report_results


def execute():
    """Execute a simple pipeline as an example."""
    # 1. Receive inputs as you want
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=float, required=True)
    inputs = parser.parse_args()

    raise RuntimeError

if __name__ == "__main__":
    execute()
