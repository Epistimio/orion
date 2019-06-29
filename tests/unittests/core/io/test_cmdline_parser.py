#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.cmdline_parser`."""
from collections import OrderedDict
import os

import pytest

from orion.core.io.cmdline_parser import CmdlineParser


@pytest.fixture
def basic_config():
    """Return a simple configuration"""
    config = OrderedDict()

    config['_pos_0'] = 'python'
    config['_pos_1'] = 'script.py'
    config['_pos_2'] = 'some'
    config['_pos_3'] = 'pos'
    config['_pos_4'] = 'args'
    config['with'] = 'args'
    config['and'] = ['multiple', 'args']
    config['plus'] = True
    config['booleans'] = True
    config['equal'] = 'value'

    return config


@pytest.fixture
def to_format():
    """Return a commandline to format"""
    return "python 1 --arg value --args value1 value2 --boolean"


def test_key_to_arg():
    """Test the key to arg function"""
    cmdline_parser = CmdlineParser()
    assert cmdline_parser._key_to_arg("c") == "-c"
    assert cmdline_parser._key_to_arg("test") == "--test"
    assert cmdline_parser._key_to_arg("test-test") == "--test-test"


def test_parse_paths(monkeypatch):
    """Test the parse_paths function"""
    monkeypatch.chdir(os.path.dirname(__file__))
    cmdline_parser = CmdlineParser()
    assert cmdline_parser._parse_paths(__file__) == os.path.abspath(__file__)

    values = ['test_resolve_config.py', 'test', 'nada', __file__]
    parsed_values = cmdline_parser._parse_paths(values)
    assert parsed_values[0] == os.path.abspath('test_resolve_config.py')
    assert parsed_values[1] == 'test'
    assert parsed_values[2] == 'nada'
    assert parsed_values[3] == os.path.abspath(__file__)


def test_parse_arguments(basic_config):
    """Test the parsing of the commandline arguments"""
    cmdline_parser = CmdlineParser()
    cmdline_parser._parse_arguments(
        "python script.py some pos args --with args --and multiple args "
        "--plus --booleans --equal=value".split(" "))

    assert cmdline_parser.arguments == basic_config


def test_parse_arguments_template():
    """Test the creation of the template for arguments"""
    cmdline_parser = CmdlineParser()

    cmdline_parser.parse(
        "python script.py some pos args "
        "--with args --and multiple args --plus --booleans --equal=value".split(" "))

    assert (
        cmdline_parser.template ==
        ['{_pos_0}', '{_pos_1}', '{_pos_2}', '{_pos_3}', '{_pos_4}', '--with', '{with}', '--and',
         '{and[0]}', '{and[1]}', '--plus', '--booleans', '--equal', '{equal}'])


def test_format(to_format):
    """Test that the format method assigns the correc values"""
    cmdline_parser = CmdlineParser()

    cmdline_parser.parse(to_format.split(' '))

    formatted = cmdline_parser.format(cmdline_parser.arguments)

    assert formatted == to_format.split(' ')


def test_parse_arguments_bad_command():
    """Test the fail cases of parsing"""
    cmdline_parser = CmdlineParser()

    with pytest.raises(ValueError) as exc_info:
        cmdline_parser.parse(
            "python script.py some pos args "
            "--with args --and multiple args --plus --booleans "
            "--and dummy.yaml".split(" "))

    assert "Conflict: two arguments have the same name: and" in str(exc_info.value)


def test_has_already_been_parsed():
    """Test the template from the branching"""
    cmdline_parser = CmdlineParser()

    command = "python script.py some pos args " \
              "--with args --and multiple args --plus --booleans "

    cmdline_parser.parse(command.split(' '))
    with pytest.raises(RuntimeError) as exc_info:
        cmdline_parser.parse(command.split(' '))

    assert "already" in str(exc_info.value)
