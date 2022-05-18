#!/usr/bin/env python
"""Simple one dimensional example for a possible user's script."""
import argparse
import os

from orion.client import report_results


def function(x):
    """Evaluate partial information of a quadratic."""
    z = x - 34.56789
    return 4 * z**2 + 23.4, 8 * z


def execute():
    """Execute a simple pipeline as an example."""
    # 1. Receive inputs as you want
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=float, required=True)
    parser.add_argument("--test-env", action="store_true")
    parser.add_argument("--experiment-id", type=str)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--experiment-version", type=str)
    parser.add_argument("--trial-id", type=str)
    parser.add_argument("--working-dir", type=str)

    inputs = parser.parse_args()

    if inputs.test_env:
        assert inputs.experiment_id == os.environ["ORION_EXPERIMENT_ID"]
        assert inputs.experiment_name == os.environ["ORION_EXPERIMENT_NAME"]
        assert inputs.experiment_version == os.environ["ORION_EXPERIMENT_VERSION"]
        assert inputs.trial_id == os.environ["ORION_TRIAL_ID"]
        assert inputs.working_dir == os.environ["ORION_WORKING_DIR"]

    # 2. Perform computations
    y, dy = function(inputs.x)

    # 3. Gather and report results
    results = list()
    results.append(dict(name="example_objective", type="objective", value=y))
    results.append(dict(name="example_gradient", type="gradient", value=[dy]))

    report_results(results)


if __name__ == "__main__":
    execute()
