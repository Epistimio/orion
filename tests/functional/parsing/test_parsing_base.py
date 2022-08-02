#!/usr/bin/env python
"""Perform functional tests for the parsing of the basic arguments groups."""
import argparse
import os

import orion.core.cli.base as cli


def _create_parser(need_subparser=True):
    parser = argparse.ArgumentParser()

    if need_subparser:
        subparsers = parser.add_subparsers()
        return parser, subparsers

    return parser


def test_common_group_arguments(orionstate, monkeypatch):
    """Check the parsing of the common group"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser, subparsers = _create_parser()
    args_list = ["-n", "test", "--config", "./orion_config_random.yaml"]

    cli.get_basic_args_group(parser)
    args = vars(parser.parse_args(args_list))
    assert args["name"] == "test"
    assert args["config"].name == "./orion_config_random.yaml"


def test_user_group_arguments(orionstate, monkeypatch):
    """Test the parsing of the user group"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = _create_parser(False)
    args_list = ["./black_box.py", "-x~normal(50,50)"]

    cli.get_user_args_group(parser)
    args = vars(parser.parse_args(args_list))
    assert len(args["user_args"]) == 2
    assert args["user_args"] == ["./black_box.py", "-x~normal(50,50)"]


def test_common_and_user_group_arguments(orionstate, monkeypatch):
    """Test the parsing of the command and user groups"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = _create_parser(False)
    args_list = [
        "-n",
        "test",
        "-c",
        "./orion_config_random.yaml",
        "./black_box.py",
        "-x~normal(50,50)",
    ]

    cli.get_basic_args_group(parser)
    cli.get_user_args_group(parser)
    args = vars(parser.parse_args(args_list))
    assert args["name"] == "test"
    assert args["config"].name == "./orion_config_random.yaml"
    assert len(args["user_args"]) == 2
    assert args["user_args"] == ["./black_box.py", "-x~normal(50,50)"]
