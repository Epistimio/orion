#!/usr/bin/env python
"""Simple script simulating behaviour of a program that requires inputs
to determine and interact with a unique directory in local disk.
"""
import argparse
import os

from orion.client import report_results


def function(x):
    """Evaluate partial information of a quadratic."""
    z = x - 34.56789168765984988448213179176
    return 4 * z**2 + 23.4, 8 * z


def execute():
    """Execute a simple pipeline as an example."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=float, required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--other-name", type=str, required=True)
    inputs = parser.parse_args()

    # That's what is expected to happen
    os.makedirs(
        os.path.join(inputs.dir, inputs.other_name, f"my-exp-{inputs.name}"),
        exist_ok=False,
    )  # Raise OSError if it exists

    y, dy = function(inputs.x)

    results = list()
    results.append(dict(name="example_objective", type="objective", value=y))
    results.append(dict(name="example_gradient", type="gradient", value=[dy]))

    report_results(results)


if __name__ == "__main__":
    execute()
