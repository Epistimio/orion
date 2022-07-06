#!/usr/bin/env python
"""Simple one dimensional example for a possible user's script."""

import argparse

from orion.client import report_results


def function(x):
    """Evaluate partial information of a quadratic."""
    y = x - 34.56789
    return 4 * y**2 + 23.4, 8 * y


def execute():
    """Execute a simple pipeline as an example."""
    # 1. Receive inputs as you want
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=float, required=True)
    inputs = parser.parse_args()

    # 2. Perform computations
    y, dy = function(inputs.x)

    # 3. Gather and report results
    results = list()
    results.append(dict(name="example_objective", type="objective", value=y))
    results.append(dict(name="example_gradient", type="gradient", value=[dy]))

    report_results(results)


if __name__ == "__main__":
    execute()
