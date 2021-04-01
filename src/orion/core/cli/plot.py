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

log = logging.getLogger(__name__)
DESCRIPTION = "Produce plots for Oríon experiments"


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    plot_parser = parser.add_parser("plot", help=DESCRIPTION, description=DESCRIPTION)

    cli.get_basic_args_group(plot_parser)

    plot_parser.add_argument(
        "kind",
        type=str,
        choices=['lpi', 'partial_dependencies', 'parallel_coordinates', 'regret'],
        help="kind of plot to generate. "
        " Pick one among ['lpi', 'partial_dependencies', 'parallel_coordinates', 'regret']"
    )

    plot_parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="png",
        choices=['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'],
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
        " Scale acts as multiplier on height and width of resulting image."
        " Overrides value of 'scale' in plotly.io.write_image."
    )

    plot_parser.set_defaults(func=main)

    return plot_parser


def main(args):
    """Starts an application that will generate a plot."""
    
    # Note : If you specify no argument at all (except 'kind'),
    #        the default behavior is to plot "{experiment.name}_{kind}.png".
    
    experiment = experiment_builder.build_from_args(args)
    output_plot = experiment.plot(kind=args['kind'])

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
