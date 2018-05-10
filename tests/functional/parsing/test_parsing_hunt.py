#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform functional tests for the parsing of the `hunt` command."""
import argparse
import os

import pytest

from orion.core.cli import hunt


def _create_parser(need_subparser=True):
    parser = argparse.ArgumentParser()

    if need_subparser:
        subparsers = parser.add_subparsers()
        return parser, subparsers

    return parser


@pytest.mark.usefixtures("clean_db")
def test_hunt_command_full_parsing(database, monkeypatch):
    """Test the parsing of the `hunt` command"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser, subparsers = _create_parser()
    args_list = ["hunt", "-n", "test",
                 "--config", "./orion_config_random.yaml",
                 "--max-trials", "400", "--pool-size", "4",
                 "./black_box.py", "-x~normal(1,1)"]

    hunt.get_parser(subparsers)
    subparsers.choices['hunt'].set_defaults(func='')

    args = vars(parser.parse_args(args_list))
    assert args['name'] == 'test'
    assert args['config'].name == './orion_config_random.yaml'
    assert args['user_script'] == './black_box.py'
    assert args['user_args'] == ['-x~normal(1,1)']
    assert args['pool_size'] == 4
    assert args['max_trials'] == 400
