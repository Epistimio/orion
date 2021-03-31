#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module running the plot command
========================

Exposes the interface for plotting for command-line usage.

"""
import argparse
import logging

import orion.core.io.experiment_builder as experiment_builder
from orion.core.cli import base as cli
from orion.storage.base import get_storage
import orion.plotting.base

# c'est quoi que ça fait ça?
#from orion.storage.base import get_storage


log = logging.getLogger(__name__)
DESCRIPTION = "Produce plots for Oríon experiments"



def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    plot_parser = parser.add_parser("plot", help=DESCRIPTION, description=DESCRIPTION)

    cli.get_basic_args_group(plot_parser)

    plot_parser.add_argument(
        "kind",
        type=str,
        help="kind of plot to generate. "
        " Pick one among ['lpi', 'partial_dependencies', 'parallel_coordinates', 'regret']"
    )

    plot_parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="png",
        help="type of plot to return. "
        " Pick one among ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json']"
        " (default: png)",
    )

    plot_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="path where plot is saved. "
        " Will override the `type` argument."
        " (default is {exp.name}_{kind}.{type})"
    )

    plot_parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="more pixels, but same proportions of the plot. "
        " Reference is 1.0."
        " Overrides value of 'scale' in plotly.io.write_image."
    )

    plot_parser.set_defaults(func=main)

    return plot_parser


def main(args):
    """Starts an application that will generate a plot."""
    config = experiment_builder.get_cmd_config(args)

    # Note : If you specify no argument at all, the default behavior
    #        is to plot the lpi as <experiment_name>.png .


    # What does this do?    
    #experiment_builder.setup_storage(config.get("storage"))
    
    experiment = experiment_builder.build_from_args(args)

    assert args['kind'] in ['lpi', 'partial_dependencies', 'parallel_coordinates', 'regret']
    func_plotting = getattr(orion.plotting.base, args['kind'])

    output_plot = func_plotting(experiment)

    valid_types = ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json']
    # 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf' handled by fig.write_image
    # 'html' handled by fig.write_html
    # 'json' handled by just writing out the file

    if args['output']:
        # `output` was given, and we'll override `type`
        args['type'] = args['output'].split(".")[-1]
        assert args['type'] in valid_types, (
            "Overriding the `type` field with something from `output`, "
            f"but we got the invalid value : {args['type']} .")
    else:
        # `type` was given, and we'll pick a `output` based on the name
        assert args['type'] in valid_types, (
            f"Invalid `type` {args['type']} .")
        # Using `experiment.name` instead of args['name'] because it leaves
        # orion the possibility of having inferred the name in other ways.
        args['output'] = f"{experiment.name}_{args['kind']}.{args['type']}"

    if args['type'] in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        # a way to increase pixel counts without affecting the proportions
        kwargs = {}
        if args['scale'] is not None:
            kwargs['scale'] = args['scale']

        output_plot.write_image(args['output'], **kwargs)
    elif args['type'] in ['html']:
        output_plot.write_html(args['output'])
    elif args['type'] in ['json']:
        with open(args['output'], "w") as f_out:
            # Note that this is the content of the "body" in the WebApi.
            f_out.write(output_plot.to_json())
    else:
        raise Exception("This is a bug. You should never land here if the logic is not faulty.")
