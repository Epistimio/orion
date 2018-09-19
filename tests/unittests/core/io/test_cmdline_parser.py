#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.cmdline_parser`."""
from collections import OrderedDict
import os

import pytest

from orion.core.io.cmdline_parser import CmdlineParser


@pytest.fixture
def basic_config():
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
    return config


def test_arg_to_key():
    cmdline_parser = CmdlineParser()
    assert cmdline_parser._arg_to_key("-c") == "c"
    assert cmdline_parser._arg_to_key("--test") == "test"
    assert cmdline_parser._arg_to_key("--test.test") == "test.test"
    assert cmdline_parser._arg_to_key("--test_test") == "test!!test"
    assert cmdline_parser._arg_to_key("--test__test") == "test!!!!test"
    assert cmdline_parser._arg_to_key("--test-test") == "test??test"
    assert cmdline_parser._arg_to_key("--test-some=thing") == "test??some"
    assert cmdline_parser._arg_to_key("--test.some=thing") == "test.some"
    assert cmdline_parser._arg_to_key("--test-some=thing=is=weird") == "test??some"


def test_bad_arg_to_key():
    cmdline_parser = CmdlineParser()
    with pytest.raises(ValueError):
        assert cmdline_parser._arg_to_key("-c-c")

    with pytest.raises(ValueError):
        assert cmdline_parser._arg_to_key("--c")

    with pytest.raises(ValueError):
        assert cmdline_parser._arg_to_key("--c")


def test_key_to_arg():
    cmdline_parser = CmdlineParser()
    assert cmdline_parser._key_to_arg("c") == "-c"
    assert cmdline_parser._key_to_arg("test") == "--test"
    assert cmdline_parser._key_to_arg("test.test") == "--test.test"
    assert cmdline_parser._key_to_arg("test!!test") == "--test_test"
    assert cmdline_parser._key_to_arg("test!!!!test") == "--test__test"
    assert cmdline_parser._key_to_arg("test??test") == "--test-test"
    assert cmdline_parser._key_to_arg("test??some") == "--test-some"
    assert cmdline_parser._key_to_arg("test.some") == "--test.some"
    assert cmdline_parser._key_to_arg("test!!some") == "--test_some"


def test_parse_paths(monkeypatch):
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
    cmdline_parser = CmdlineParser()
    cmdline_parser._parse_arguments(
        "python script.py some pos args "
        "--with args --and multiple args --plus --booleans".split(" "))

    assert cmdline_parser.arguments == basic_config


def test_parse_arguments_configuration(basic_config):
    cmdline_parser = CmdlineParser()

    configuration = cmdline_parser.parse(
        "python script.py some pos args "
        "--with args --and multiple args --plus --booleans".split(" "))
    assert configuration == basic_config


def test_parse_arguments_template():
    cmdline_parser = CmdlineParser()

    cmdline_parser.parse(
        "python script.py some pos args "
        "--with args --and multiple args --plus --booleans".split(" "))

    assert (
        cmdline_parser.template ==
        ['{_pos_0}', '{_pos_1}', '{_pos_2}', '{_pos_3}', '{_pos_4}', '--with', '{with}', '--and',
         '{and[0]}', '{and[1]}', '--plus', '--booleans'])


def test_parse_arguments_bad_command():
    cmdline_parser = CmdlineParser()

    with pytest.raises(ValueError) as exc_info:
        cmdline_parser.parse(
            "python script.py some pos args "
            "--with args --and multiple args --plus --booleans "
            "--and dummy.yaml".split(" "))

    assert "Two arguments have the same name: and" in str(exc_info.value)


def test_parse_branching_arguments_template():
    cmdline_parser = CmdlineParser()

    command = ("python script.py some pos args "
               "--with args --and multiple args --plus --booleans ")

    cmdline_parser.parse(command.split(" "))
    assert (
        cmdline_parser.template ==
        ['{_pos_0}', '{_pos_1}', '{_pos_2}', '{_pos_3}', '{_pos_4}', '--with', '{with}', '--and',
         '{and[0]}', '{and[1]}', '--plus', '--booleans'])

    branch_configuration = cmdline_parser.parse("--with something --to update".split(" "))

    assert branch_configuration == {
        'with': 'something',
        'to': 'update'}

    assert (
        cmdline_parser.template ==
        ['{_pos_0}', '{_pos_1}', '{_pos_2}', '{_pos_3}', '{_pos_4}', '--with', '{with}', '--and',
         '{and[0]}', '{and[1]}', '--plus', '--booleans', '--to', '{to}'])


def test_parse_branching_arguments_format(monkeypatch):
    monkeypatch.chdir(os.path.dirname(__file__))

    cmdline_parser = CmdlineParser()

    command = ("python script.py some pos args "
               "--with args --and multiple args --plus --booleans ")

    configuration = cmdline_parser.parse(command.split(" "))
    print(cmdline_parser.template)
    assert cmdline_parser.format(configuration) == command.strip(" ")

    branch_configuration = cmdline_parser.parse("--with something --to update".split(" "))
    print(configuration)
    print(cmdline_parser.template)
    print(branch_configuration)
    configuration.update(branch_configuration)

    assert (
        cmdline_parser.format(configuration) ==
        command.replace("--with args", "--with something").strip(" ") + " --to update")
