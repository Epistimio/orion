#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple script simulating behaviour of a program that requires inputs
to determine and interact with a unique directory in local disk.
"""
import argparse
import os

from orion.client import report_results


def execute():
    """Execute a simple pipeline as an example."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=float, required=True)
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--other-name', type=str, required=True)
    inputs = parser.parse_args()

    # That's what is expected to happen
    os.makedirs(os.path.join(inputs.dir, inputs.other_name, "my-exp-{}".format(inputs.name)),
                exist_ok=False)  # Raise OSError if it exists

    results = list()
    results.append(dict(
        name='from_{}'.format(inputs.name),
        type='objective',
        value=66))
    results.append(dict(
        name='example_gradient',
        type='gradient',
        value=[1]))
    report_results(results)


if __name__ == "__main__":
    execute()
