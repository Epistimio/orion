#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform functional tests for the parsing of the different commands."""
import os

import numpy
import pytest
import argparse

from orion.core.io.database import Database
from orion.core.cli import (hunt, insert, init_only, resolve_config)


def create_parser(need_subparser = True):
    parser = argparse.ArgumentParser()

    if need_subparser:
        subparsers = parser.add_subparsers()
        return parser, subparsers
    
    return parser

@pytest.mark.usefixtures("clean_db")
def test_common_group_arguments(database, monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser, subparsers = create_parser()
    args_list = ["-n", "test", "--config", "./orion_config_random.yaml"]

    resolve_config.get_basic_args_group(parser)
    args = vars(parser.parse_args(args_list))
    assert args['name'] == 'test'
    assert args['config'].name == "./orion_config_random.yaml"

@pytest.mark.usefixtures("clean_db")
def test_user_group_arguments(database, monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = create_parser(False)
    args_list = ["./black_box.py", "-x~normal(50,50)"]

    resolve_config.get_user_args_group(parser)
    args = vars(parser.parse_args(args_list))
    assert args['user_script'] == './black_box.py'
    assert len(args['user_args']) == 1
    assert args['user_args'] == ['-x~normal(50,50)']

@pytest.mark.usefixtures("clean_db")
def test_common_and_user_group_arguments(database, monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = create_parser(False)
    args_list = ["-n", "test", "-c", "./orion_config_random.yaml", "./black_box.py", "-x~normal(50,50)"]

    resolve_config.get_basic_args_group(parser)
    resolve_config.get_user_args_group(parser)
    args = vars(parser.parse_args(args_list))
    assert args['name'] == 'test'
    assert args['config'].name == './orion_config_random.yaml'
    assert args['user_script'] == './black_box.py'
    assert len(args['user_args']) == 1
    assert args['user_args'] == ['-x~normal(50,50)']


@pytest.mark.usefixtures("clean_db")
def test_hunt_command_full_parsing(database, monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser, subparsers = create_parser()
    args_list = ["hunt", "-n", "test", "--config", "./orion_config_random.yaml", "--max-trials", "400", "--pool-size", "4" , "./black_box.py", "-x~normal(1,1)"]
    
    hunt.get_parser(subparsers)
    subparsers.choices['hunt'].set_defaults(func='')

    args = vars(parser.parse_args(args_list))
    assert args['name'] == 'test'
    assert args['config'].name == './orion_config_random.yaml'
    assert args['user_script'] == './black_box.py'
    assert args['user_args'] == ['-x~normal(1,1)']
    assert args['pool_size'] == 4
    assert args['max_trials'] == 400

@pytest.mark.usefixtures("clean_db")
def test_insert_command_full_parsing(database, monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser, subparsers = create_parser()
    args_list = ["insert", "-n", "test", "--config", "./orion_config_random.yaml", "./black_box.py", "-x=1"]
    
    insert.get_parser(subparsers)
    subparsers.choices['insert'].set_defaults(func='')

    args = vars(parser.parse_args(args_list))
    assert args['name'] == 'test'
    assert args['config'].name == './orion_config_random.yaml'
    assert args['user_script'] == './black_box.py'
    assert args['user_args'] == ['-x=1']

@pytest.mark.usefixtures("clean_db")
def test_init_only_command_full_parsing(database, monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser, subparsers = create_parser()
    args_list = ["init_only", "-n", "test", "--config", "./orion_config_random.yaml", "./black_box.py", "-x~normal(1,1)"]
    
    init_only.get_parser(subparsers)
    subparsers.choices['init_only'].set_defaults(func='')

    args = vars(parser.parse_args(args_list))
    assert args['name'] == 'test'
    assert args['config'].name == './orion_config_random.yaml'
    assert args['user_script'] == './black_box.py'
    assert args['user_args'] == ['-x~normal(1,1)']
