#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple one dimensional example for a possible user's script."""
import argparse
import time

from orion.client import report_results


def function(x):
    """Evaluate partial information of a quadratic."""
    z = x - 34.56789
    return 4 * z**2 + 23.4


def execute():
    """Execute a simple pipeline as an example."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=float, required=True)
    inputs = parser.parse_args()

    y = function(inputs.x)

    time.sleep(1)
    results = list()
    results.append(dict(
        name='example_objective',
        type='objective',
        value=y))
    report_results(results)


if __name__ == "__main__":
    execute()
