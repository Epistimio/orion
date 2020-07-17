#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple one dimensional example for a possible user's script."""
import argparse

import numpy

from orion.client import report_results


def rosenbrock_function(x):
    """Evaluate a n-D rosenbrock function."""
    x = numpy.asarray(x)
    summands = 100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2
    return numpy.sum(summands)


def execute():
    """Execute a simple pipeline as an example."""
    # 1. Receive inputs as you want
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', type=str, required=True,
                        help="Representation of a list of floating numbers of "
                             "length at least 2.")
    inputs = parser.parse_args()

    # 2. Perform computations
    x = numpy.fromstring(inputs.x[1:-1], sep=', ')
    f = rosenbrock_function(x)

    # 3. Gather and report results
    results = list()
    results.append(dict(
        name='rosenbrock',
        type='objective',
        value=f))
    report_results(results)


if __name__ == "__main__":
    execute()
