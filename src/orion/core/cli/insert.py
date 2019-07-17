#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=eval-used,protected-access
"""
:mod:`orion.core.cli.insert` -- Module to insert new trials
===========================================================

.. module:: insert
   :platform: Unix
   :synopsis: Insert creates new trials for a given experiment with fixed values

"""
import collections
import logging
import os
import re

from orion.core.cli import base as cli
from orion.core.io.convert import infer_converter_from_file_type
from orion.core.io.experiment_builder import ExperimentBuilder
from orion.core.utils.format_trials import tuple_to_trial

log = logging.getLogger(__name__)


def add_subparser(parser):
    """Add the subparser that needs to be used for this command"""
    insert_parser = parser.add_parser('insert', help='insert help')

    cli.get_basic_args_group(insert_parser)

    cli.get_user_args_group(insert_parser)

    insert_parser.set_defaults(func=main)

    return insert_parser


def main(args):
    """Fetch config and insert new point"""
    command_line_user_args = args.pop('user_args', [None])[1:]
    # TODO: Views are not fully configured until configuration is refactored
    experiment = ExperimentBuilder().build_view_from(args)
    # TODO: Remove this line when views gets fully configured
    experiment = ExperimentBuilder().build_from(args)

    transformed_args = _build_from(command_line_user_args)
    exp_space = experiment.space

    values = _create_tuple_from_values(transformed_args, exp_space)

    trial = tuple_to_trial(values, exp_space)

    experiment.register_trial(trial)


def _validate_dimensions(transformed_args, exp_space):
    exp_namespaces = list(exp_space.keys())
    values = {}

    # Find any dimension that is not given by the user and make sure they have a default value
    for exp_n in exp_namespaces:
        if exp_n not in transformed_args.keys() and exp_space[exp_n].default_value is None:
            error_msg = "Dimension {} is unspecified and has no default value".format(exp_n)
            raise ValueError(error_msg)

    # Find any namespace that is not in the space of the experiment,
    # or values that lie outside the prior's interval
    for namespace, value in transformed_args.items():
        if namespace not in exp_space:
            error_msg = "Found namespace outside of experiment space : {}".format(namespace)
            raise ValueError(error_msg)

        valid, value = _validate_input_value(value, exp_space, namespace)

        if not valid:
            error_msg = "Value {} is outside of dimension's prior interval".format(value)
            raise ValueError(error_msg)

        values[namespace] = value

    return values


def _validate_input_value(value, exp_space, namespace):

    is_valid = False

    try:
        value = eval(value)
        is_valid = value in exp_space[namespace]
    except NameError:
        is_valid = value in exp_space[namespace]

    return is_valid, value


def _create_tuple_from_values(transformed_args, exp_space):
    values_dict = _validate_dimensions(transformed_args, exp_space)
    values = []

    for namespace in exp_space:
        if namespace in values_dict.keys():
            values.append(values_dict[namespace])
        else:
            values.append(exp_space[namespace].default_value)

    return values


def _build_from(cmd_args):
    transformed_args, userconfig, _ = _build_from_args(cmd_args)

    if userconfig:
        transformed_args = _build_from_config(userconfig)

    return transformed_args


def _build_from_config(config_path):
    converter = infer_converter_from_file_type(config_path)
    userconfig_tmpl = converter.parse(config_path)
    userconfig_keyword = 'orion='

    transformed_args = {}
    stack = collections.deque()
    stack.append(('', userconfig_tmpl))
    while True:
        try:
            namespace, stuff = stack.pop()
        except IndexError:
            break
        if isinstance(stuff, dict):
            for k, v in stuff.items():
                stack.append(('/'.join([namespace, str(k)]), v))
        elif isinstance(stuff, list):
            for position, thing in enumerate(stuff):
                stack.append(('/'.join([namespace, str(position)]), thing))
        elif isinstance(stuff, str):
            if stuff.startswith(userconfig_keyword):
                if namespace in transformed_args:
                    error_msg = "Conflict for name '{}' in script configuration "\
                                "and arguments.".format(namespace)
                    raise ValueError(error_msg)

                transformed_args[namespace] = stuff[len(userconfig_keyword):]

    return transformed_args


def _build_from_args(cmd_args):
    userconfig = None
    is_userconfig_an_option = None
    userargs_search = r'\W*([a-zA-Z0-9_-]+)=([^ ]+)'
    userargs_regex_tmpl = r'(.*)=(.*)'
    userconfig_option = "--config="

    transformed_args = {}
    userargs_tmpl = collections.defaultdict(list)
    args_pattern = re.compile(userargs_search)
    args_prefix_pattern = re.compile(userargs_regex_tmpl)

    for arg in cmd_args:
        found = args_pattern.findall(arg)
        if len(found) != 1:
            if arg.startswith(userconfig_option):
                if not userconfig:
                    userconfig = arg[len(userconfig_option):]
                    is_userconfig_an_option = True
                else:
                    raise ValueError(
                        "Already found one configuration file in: %s" %
                        userconfig
                        )
            else:
                userargs_tmpl[None].append(arg)
            continue

        name, value = found[0]
        namespace = '/' + name

        if namespace in transformed_args:
            error_msg = "Conflict for name '{}' in script configuration "\
                        "and arguments.".format(namespace)
            raise ValueError(error_msg)

        transformed_args[namespace] = value

        found = args_prefix_pattern.findall(arg)
        assert len(found) == 1 and found[0][1] == value, "Parsing prefix problem."
        userargs_tmpl[namespace] = found[0][0] + '='

    if not userconfig and userargs_tmpl[None]:  # try the first positional argument
        if os.path.isfile(userargs_tmpl[None][0]):
            userconfig = userargs_tmpl[None].pop(0)
            is_userconfig_an_option = False

    return transformed_args, userconfig, is_userconfig_an_option
