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
        "-r",
        "--resource",
        type=str,
        default="lpi",
        help="resource to plot;"
        " Pick one among ['lpi', 'partial_dependencies', 'parallel_coordinates', 'regret']"
        " (default: lpi)",
    )

    plot_parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="png",
        help="type of plot to return;"
        " Pick one among ['png', 'jpeg', 'json']"
        " (default: png)",
    )

    plot_parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="",
        help="path where plot is saved;"
        " Will override the `type` argument."
        " (default is experiment name if applicable)"
    )

    plot_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="dpi when plotting png;"
        " (default: 150)",
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

    assert args['resource'] in ['lpi', 'partial_dependencies', 'parallel_coordinates', 'regret']
    func_plotting = getattr(orion.plotting.base, args['resource'])

    output_plot = func_plotting(experiment)

    valid_types = ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json']
    # 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf' handled by fig.write_image
    # 'html' handled by fig.write_html
    # 'json' handled by just writing out the file

    if args['output_path']:
        # `output_path` was given, and we'll override `type`
        args['type'] = args['output_path'].split(".")[-1]
        assert args['type'] in valid_types, (
            "Overriding the `type` field with something from `output_path`, "
            f"but we got the invalid value : {args['type']} .")
    else:
        # `type` was given, and we'll pick a `output_path` based on the name
        assert args['type'] in valid_types, (
            f"Invalid `type` {args['type']} .")
        assert args["name"], "No name found in arguments."
        args['output_path'] = args['name'] + "." + args['type']

    if args['type'] in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        # TODO : Use dpi in png.
        output_plot.write_image(args['output_path'])
    elif args['type'] in ['html']:
        output_plot.write_html(args['output_path'])
    elif args['type'] in ['json']:
        with open(args['output_path'], "w") as f_out:
            # Note that this is the content of the "body" in the WebApi.
            f_out.write(output_plot.to_json())
    else:
        raise Exception("This is a bug. You should never land here if the logic is not faulty.")
