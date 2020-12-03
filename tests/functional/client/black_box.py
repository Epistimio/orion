#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script that will always interrupt trials."""
import argparse

from orion.client import (  # noqa: F401
    interrupt_trial,
    report_bad_trial,
    report_objective,
)


def no_report():
    """Do not report any result"""
    return


def execute():
    """Execute a simple pipeline as an example."""
    parser = argparse.ArgumentParser()
    parser.add_argument("fct", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("-x", type=float)

    inputs = parser.parse_args()

    kwargs = {}

    # Maybe its a float, maybe user made a mistake and report objective='name'
    try:
        inputs.objective = float(inputs.objective)
    except (ValueError, TypeError):
        pass

    for key, value in vars(inputs).items():
        if value is not None:
            kwargs[key] = value

    kwargs.pop("fct")
    kwargs.pop("x")

    if "data" in kwargs:
        kwargs["data"] = [dict(name=kwargs["data"], type="constraint", value=1.0)]

    globals()[inputs.fct](**kwargs)


if __name__ == "__main__":
    execute()
