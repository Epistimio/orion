#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform functional tests for the parsing of the different commands."""
import argparse
import os

import pytest

from orion.core.cli import insert


def _create_parser(need_subparser=True):
    parser = argparse.ArgumentParser()

    if need_subparser:
        subparsers = parser.add_subparsers()
        return parser, subparsers

    return parser


@pytest.mark.usefixtures("clean_db")
def test_insert_command_full_parsing(database, monkeypatch):
    """Test the parsing of all the options of insert"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser, subparsers = _create_parser()
    args_list = ["insert", "-n", "test", "--config",
                 "./orion_config_random.yaml", "./black_box.py", "-x=1"]

    insert.get_parser(subparsers)
    subparsers.choices['insert'].set_defaults(func='')

    args = vars(parser.parse_args(args_list))
    assert args['name'] == 'test'
    assert args['config'].name == './orion_config_random.yaml'
    assert args['user_script'] == './black_box.py'
    assert args['user_args'] == ['-x=1']
