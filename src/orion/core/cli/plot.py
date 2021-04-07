#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module running the plot command
========================

Exposes the interface for plotting for command-line usage.

"""
import logging

import orion.core.io.experiment_builder as experiment_builder
from orion.client.experiment import ExperimentClient
from orion.core.cli import base as cli
from orion.plotting.base import SINGLE_EXPERIMENT_PLOTS

log = logging.getLogger(__name__)
DESCRIPTION = "Produce plots for Oríon experiments"


IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "svg", "pdf"]
HTML_TYPES = ["html"]
JSON_TYPES = ["json"]
VALID_TYPES = IMAGE_TYPES + HTML_TYPES + JSON_TYPES


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    plot_parser = parser.add_parser("plot", help=DESCRIPTION, description=DESCRIPTION)

    cli.get_basic_args_group(plot_parser)

    plot_parser.add_argument(
        "kind",
        type=str,
        choices=SINGLE_EXPERIMENT_PLOTS.keys(),
        help="kind of plot to generate. ",
    )

    plot_parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="png",
        choices=VALID_TYPES,
        help="type of plot to return. (default: png)",
    )

    plot_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="path where plot is saved. "
        " Will override the `type` argument."
        " (default is {exp.name}-v{exp.version}_{kind}.{type})",
    )

    plot_parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="more pixels, but same proportions of the plot. "
        " Scale acts as multiplier on height and width of resulting image."
        " Overrides value of 'scale' in plotly.io.write_image.",
    )

    plot_parser.set_defaults(func=main)

    return plot_parser


def infer_type(output, out_type):
    """Infer type of plot file based on output filename or provided type.

    If output has a valid extension, this extension is used as the type. Otherwise,
    the provided (or default) output type is used.
    """
    if output:
        ext = output.split(".")[-1]
        if ext and ext not in VALID_TYPES:
            log.warning(
                "Overriding the `type` field with something from `output`, "
                f"but we got the invalid value : {out_type}. Will revert to png."
                f"Should be one of {VALID_TYPES}"
            )
        else:
            out_type = ext

    return out_type


def get_output(experiment, output, kind, out_type):
    """Create output file name based on experiment name, plot kind and file type.

    If the output filename is provided, it is appended with the file type if filename
    does not already has the corresponding extention. (ex output.name -> output.name.png)
    """

    if not output:
        return f"{experiment.name}-v{experiment.version}_{kind}.{out_type}"
    elif not output.endswith(f".{out_type}"):
        return f"{output}.{out_type}"

    return output


def main(args):
    """Starts an application that will generate a plot."""

    # Note : If you specify no argument at all (except 'kind'),
    #        the default behavior is to plot "{experiment.name}_{kind}.png".

    experiment = ExperimentClient(
        experiment_builder.get_from_args(args, mode="r"), None
    )
    output_plot = experiment.plot(kind=args["kind"])

    args["type"] = infer_type(args["output"], args["type"])
    args["output"] = get_output(experiment, args["output"], args["kind"], args["type"])

    if args["type"] in IMAGE_TYPES:
        output_plot.write_image(args["output"], scale=args["scale"])
    elif args["type"] in HTML_TYPES:
        output_plot.write_html(args["output"])
    elif args["type"] in JSON_TYPES:
        with open(args["output"], "w") as f_out:
            # Note that this is the content of the "body" in the WebApi.
            f_out.write(output_plot.to_json())
    else:
        raise Exception(
            "This is a bug. You should never land here if the logic is not faulty."
        )
