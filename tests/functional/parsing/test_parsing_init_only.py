#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform functional tests for the parsing of `init_only` command."""
import argparse
import os

import pytest

from orion.core.cli import init_only


def _create_parser(need_subparser=True):
    parser = argparse.ArgumentParser()

    if need_subparser:
        subparsers = parser.add_subparsers()
        return parser, subparsers

    return parser


@pytest.mark.usefixtures("clean_db")
def test_init_only_command_full_parsing(database, monkeypatch):
    """Make sure the parsing of init_only is done correctly."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser, subparsers = _create_parser()
    args_list = ["init_only", "-n", "test", "--config",
                 "./orion_config_random.yaml", "./black_box.py",
                 "-x~normal(1,1)"]

    init_only.add_subparser(subparsers)
    subparsers.choices['init_only'].set_defaults(func='')

    args = vars(parser.parse_args(args_list))
    assert args['name'] == 'test'
    assert args['config'].name == './orion_config_random.yaml'
    assert args['user_args'] == ['./black_box.py', '-x~normal(1,1)']
