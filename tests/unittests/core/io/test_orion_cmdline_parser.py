#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.core.io.orion_cmdliner_parser`."""
import os
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Tuple

import pytest

from orion.core.io.convert import JSONConverter
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial


@pytest.fixture
def script_path():
    """Return existing script path for the command lines"""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def parser():
    """Return an instance of `OrionCmdlineParser`."""
    return OrionCmdlineParser(allow_non_existing_files=True)


@pytest.fixture
def parser_diff_prefix():
    """Return an instance of `OrionCmdlineParser` with a different config prefix."""
    return OrionCmdlineParser(config_prefix="config2", allow_non_existing_files=True)


class WeirdArgument(NamedTuple):
    """NamedTuple that contains the fields used to construct a weird argument used in tests.

    This is nicer than having the `weird_argument` fixture return a length-4 unnamed tuple.
    """

    name: str
    prior_str: str
    prior_type: str
    value: Any


@pytest.fixture(
    params=[
        ("choices({{'a': 0.5, 'b': 0.5}})", "categorical", "a"),
        ("uniform(0, 10, discrete=True)", "integer", "2"),
        ("uniform(0, 10)", "real", "2.3"),
    ]
)
def prior_and_prior_type_and_value(request):
    """Fixture that gives a prior str, the prior type, and a value."""
    prior_str, prior_type, value = request.param
    return prior_str, prior_type, value


@pytest.fixture(
    params=[
        "name:with:colons",
        "name.with.dots",
        "name$with$dollars",
        "name/with/slashes",
        "name^with^weird",
        "name^with_weird?!^%^&*()_+_)(*&^%$%^&*()_chars",
    ]
)
def weird_argument(request, prior_and_prior_type_and_value: Tuple[str, str, str]):
    """Fixture that provides a weird name, along with a prior and value."""
    weird_param_name = request.param
    prior_str, prior_type, value = prior_and_prior_type_and_value
    return WeirdArgument(
        name=weird_param_name, prior_str=prior_str, prior_type=prior_type, value=value
    )


@pytest.fixture
def commandline(weird_argument: WeirdArgument):
    """Return a simple commandline list."""
    return [
        "--seed=555",
        "--lr~uniform(-3, 1)",
        "--non-prior=choices({{'sgd': 0.2, 'adam': 0.8}})",
        "--prior~choices({'sgd': 0.2, 'adam': 0.8})",
        "--a.b~uniform(0,1)",
        f"--{weird_argument.name}~{weird_argument.prior_str}",
    ]


@pytest.fixture
def commandline_fluff(commandline: List[str]):
    """Add extra useless info to commandline."""
    cmd_args = commandline.copy()
    cmd_args.extend(
        ["--some-path=~/some_path", "--home-path=~", "~/../folder/.hidden/folder"]
    )
    return cmd_args


@pytest.fixture
def cmd_with_properties(commandline: List[str]):
    """Add extra arguments that use `trial` and `exp` properties"""
    cmd_args = commandline.copy()
    cmd_args.extend(["--trial-name", "{trial.hash_name}", "--exp-name", "{exp.name}"])

    return cmd_args


def test_parse_from_yaml_config(parser: OrionCmdlineParser, yaml_config: List[str]):
    """Parse from a yaml config only."""
    parser.parse(yaml_config)
    config = parser.priors

    assert len(config.keys()) == 6
    assert "/layers/1/width" in config
    assert "/layers/1/type" in config
    assert "/layers/2/type" in config
    assert "/training/lr0" in config
    assert "/training/mbs" in config
    assert "/something-same" in config


def test_parse_from_json_config(parser: OrionCmdlineParser, json_config: List[str]):
    """Parse from a json config only."""
    parser.parse(json_config)
    config = parser.priors

    assert len(config.keys()) == 6
    assert "/layers/1/width" in config
    assert "/layers/1/type" in config
    assert "/layers/2/type" in config
    assert "/training/lr0" in config
    assert "/training/mbs" in config
    assert "/something-same" in config


def test_parse_from_unknown_config(
    parser: OrionCmdlineParser, some_sample_config: List[str]
):
    """Parse from a unknown config type only."""
    parser.parse(some_sample_config)
    config = parser.priors

    assert len(config.keys()) == 6
    assert "/layers/1/width" in config
    assert "/layers/1/type" in config
    assert "/layers/2/type" in config
    assert "/training/lr0" in config
    assert "/training/mbs" in config
    assert "/something-same" in config


def test_parse_equivalency(yaml_config: List[str], json_config: List[str]):
    """Templates found from json and yaml are the same."""
    parser_yaml = OrionCmdlineParser(allow_non_existing_files=True)
    parser_yaml.parse(yaml_config)
    dict_from_yaml = parser_yaml.config_file_data

    parser_json = OrionCmdlineParser(allow_non_existing_files=True)
    parser_json.parse(json_config)
    dict_from_json = parser_json.config_file_data
    assert dict_from_json == dict_from_yaml


def test_parse_from_args_only(
    parser: OrionCmdlineParser,
    commandline_fluff: List[str],
    weird_argument: WeirdArgument,
):
    """Parse a commandline."""
    cmd_args = commandline_fluff

    parser.parse(cmd_args)

    assert not parser.config_file_data
    assert len(parser.cmd_priors) == 4
    assert "/lr" in parser.cmd_priors
    assert "/prior" in parser.cmd_priors
    assert "/a.b" in parser.cmd_priors
    assert f"/{weird_argument.name}" in parser.cmd_priors

    assert parser.parser.template == [
        "--seed",
        "{seed}",
        "--lr",
        "{lr}",
        "--non-prior",
        "{non-prior}",
        "--prior",
        "{prior}",
        "--a.b",
        "{a.b}",
        f"--{weird_argument.name}",
        f"{{{weird_argument.name}}}",
        "--some-path",
        "{some-path}",
        "--home-path",
        "{home-path[0]}",
        "{home-path[1]}",
    ]


def test_parse_from_args_and_config_yaml(
    parser: OrionCmdlineParser,
    commandline: List[str],
    yaml_config: List[str],
    weird_argument: WeirdArgument,
):
    """Parse both from commandline and config file."""
    cmd_args = yaml_config
    cmd_args.extend(commandline)

    parser.parse(cmd_args)
    config = parser.priors

    assert len(config) == 10
    assert "/lr" in config
    assert "/prior" in config
    assert "/layers/1/width" in config
    assert "/layers/1/type" in config
    assert "/layers/2/type" in config
    assert "/training/lr0" in config
    assert "/training/mbs" in config
    assert "/something-same" in config
    assert "/a.b" in config

    assert f"/{weird_argument.name}" in config

    template = parser.parser.template
    assert template == [
        "--config",
        "{config}",
        "--seed",
        "{seed}",
        "--lr",
        "{lr}",
        "--non-prior",
        "{non-prior}",
        "--prior",
        "{prior}",
        "--a.b",
        "{a.b}",
        f"--{weird_argument.name}",
        f"{{{weird_argument.name}}}",
    ]


def test_parse_finds_conflict(
    parser: OrionCmdlineParser, commandline: List[str], yaml_config: List[str]
):
    """Parse find conflicting declaration in commandline and config file."""
    cmd_args = yaml_config
    cmd_args.extend(commandline)
    cmd_args.extend(["--something-same~choices({'sgd': 0.2, 'adam': 0.8})"])

    with pytest.raises(ValueError) as exc:
        parser.parse(cmd_args)

    assert "Conflict" in str(exc.value)


def test_format_commandline_only(
    parser: OrionCmdlineParser, commandline: List[str], weird_argument: WeirdArgument
):
    """Format the commandline using only args."""
    parser.parse(commandline)

    trial = Trial(
        params=[
            {"name": "/lr", "type": "real", "value": -2.4},
            {"name": "/prior", "type": "categorical", "value": "sgd"},
            {"name": "/a.b", "type": "real", "value": 0.5},
            {
                "name": f"/{weird_argument.name}",
                "type": weird_argument.prior_type,
                "value": weird_argument.value,
            },
        ]
    )

    cmd_inst = parser.format(trial=trial)
    assert cmd_inst == [
        "--seed",
        "555",
        "--lr",
        "-2.4",
        "--non-prior",
        "choices({'sgd': 0.2, 'adam': 0.8})",
        "--prior",
        "sgd",
        "--a.b",
        "0.5",
        f"--{weird_argument.name}",
        f"{weird_argument.value}",
    ]


def test_format_commandline_and_config(
    parser: OrionCmdlineParser,
    commandline: List[str],
    json_config: List[str],
    tmp_path: Path,
    json_converter,
    weird_argument: WeirdArgument,
):
    """Format the commandline and a configuration file."""
    cmd_args = json_config
    cmd_args.extend(commandline)

    parser.parse(cmd_args)
    trial = Trial(
        params=[
            {"name": "/lr", "type": "real", "value": -2.4},
            {"name": "/prior", "type": "categorical", "value": "sgd"},
            {"name": "/layers/1/width", "type": "integer", "value": 100},
            {"name": "/layers/1/type", "type": "categorical", "value": "relu"},
            {"name": "/layers/2/type", "type": "categorical", "value": "sigmoid"},
            {"name": "/training/lr0", "type": "real", "value": 0.032},
            {"name": "/training/mbs", "type": "integer", "value": 64},
            {"name": "/something-same", "type": "categorical", "value": "3"},
            {"name": "/a.b", "type": "real", "value": 0.3},
            {
                "name": f"/{weird_argument.name}",
                "type": f"{weird_argument.prior_type}",
                "value": weird_argument.value,
            },
        ]
    )

    output_file = str(tmp_path / "output.json")

    cmd_inst = parser.format(output_file, trial)

    assert cmd_inst == [
        "--config",
        output_file,
        "--seed",
        "555",
        "--lr",
        "-2.4",
        "--non-prior",
        "choices({'sgd': 0.2, 'adam': 0.8})",
        "--prior",
        "sgd",
        "--a.b",
        "0.3",
        f"--{weird_argument.name}",
        f"{weird_argument.value}",
    ]

    output_data = json_converter.parse(output_file)
    assert output_data == {
        "yo": 5,
        "training": {"lr0": 0.032, "mbs": 64},
        "layers": [
            {"width": 64, "type": "relu"},
            {"width": 100, "type": "relu"},
            {"width": 16, "type": "sigmoid"},
        ],
        "something-same": "3",
    }


def test_format_without_config_path(
    parser: OrionCmdlineParser,
    commandline: List[str],
    json_config: List[str],
    tmp_path: Path,
    json_converter: JSONConverter,
    weird_argument: WeirdArgument,
):
    """Verify that parser.format() raises ValueError when config path not passed."""
    cmd_args = json_config
    cmd_args.extend(commandline)

    parser.parse(cmd_args)

    trial = Trial(
        params=[
            {"name": "/lr", "type": "real", "value": -2.4},
            {"name": "/prior", "type": "categorical", "value": "sgd"},
            {"name": "/layers/1/width", "type": "integer", "value": 100},
            {"name": "/layers/1/type", "type": "categorical", "value": "relu"},
            {"name": "/layers/2/type", "type": "categorical", "value": "sigmoid"},
            {"name": "/training/lr0", "type": "real", "value": 0.032},
            {"name": "/training/mbs", "type": "integer", "value": 64},
            {"name": "/something-same", "type": "categorical", "value": "3"},
            {
                "name": f"/{weird_argument.name}",
                "type": "categorical",
                "value": weird_argument.value,
            },
        ]
    )

    with pytest.raises(
        ValueError, match="Cannot format without a `config_path` argument."
    ):
        parser.format(trial=trial)


def test_format_with_properties(
    parser: OrionCmdlineParser,
    cmd_with_properties: List[str],
    hacked_exp: Experiment,
    weird_argument: WeirdArgument,
):
    """Test if format correctly puts the value of `trial` and `exp` when used as properties"""
    parser.parse(cmd_with_properties)
    # NOTE: Also using a weird argument here, to make sure the parser is able to distinguish
    # property look-up vs weird argument names.

    trial = Trial(
        experiment="trial_test",
        params=[
            {"name": "/lr", "type": "real", "value": -2.4},
            {"name": "/prior", "type": "categorical", "value": "sgd"},
            {
                "name": f"/{weird_argument.name}",
                "type": weird_argument.prior_type,
                "value": weird_argument.value,
            },
        ],
    )

    cmd_line = parser.format(None, trial=trial, experiment=hacked_exp)

    assert trial.hash_name in cmd_line
    assert "supernaedo2-dendi" in cmd_line


def test_configurable_config_arg(
    parser_diff_prefix: OrionCmdlineParser, yaml_sample_path: str
):
    """Parse from a yaml config only."""
    parser_diff_prefix.parse(["--config2", yaml_sample_path])
    config = parser_diff_prefix.priors

    assert len(config.keys()) == 6
    assert "/layers/1/width" in config
    assert "/layers/1/type" in config
    assert "/layers/2/type" in config
    assert "/training/lr0" in config
    assert "/training/mbs" in config
    assert "/something-same" in config


def test_infer_user_script(script_path: str):
    """Test that user script is inferred correctly"""
    parser = OrionCmdlineParser()
    parser.parse(f"{script_path} and some args".split(" "))
    assert parser.user_script == script_path


def test_infer_user_script_python(script_path: str):
    """Test that user script is inferred correctly when using python"""
    parser = OrionCmdlineParser()
    parser.parse(f"python {script_path} and some args".split(" "))
    assert parser.user_script == script_path


def test_infer_user_script_when_missing():
    """Test that user script is inferred correctly even if missing"""
    parser = OrionCmdlineParser()

    with pytest.raises(FileNotFoundError) as exc:
        parser.parse("python script.py and some args".split(" "))
    assert exc.match("The path specified for the script does not exist")

    parser = OrionCmdlineParser(allow_non_existing_files=True)
    parser.parse("python script.py and some args".split(" "))
    assert parser.user_script == "script.py"


def test_configurable_config_arg_do_not_exist(script_path: str):
    """Test that parser can handle command if config file does not exist"""
    parser = OrionCmdlineParser()
    command = f"python {script_path} --config idontexist.yaml".split(" ")
    with pytest.raises(OSError) as exc:
        parser.parse(command)
    assert exc.match("The path specified for the script config does not exist")

    parser = OrionCmdlineParser(allow_non_existing_files=True)
    parser.parse(command)


def test_get_state_dict_before_parse(
    parser: OrionCmdlineParser, commandline: List[str]
):
    """Test getting state dict."""
    assert parser.get_state_dict() == {
        "parser": {"keys": [], "arguments": [], "template": []},
        "cmd_priors": list(map(list, parser.cmd_priors.items())),
        "file_priors": list(map(list, parser.file_priors.items())),
        "config_file_data": parser.config_file_data,
        "config_prefix": parser.config_prefix,
        "file_config_path": parser.file_config_path,
        "converter": None,
    }


def test_get_state_dict_after_parse_no_config_file(
    parser: OrionCmdlineParser, commandline: List[str]
):
    """Test getting state dict."""
    parser.parse(commandline)

    assert parser.get_state_dict() == {
        "parser": parser.parser.get_state_dict(),
        "cmd_priors": list(map(list, parser.cmd_priors.items())),
        "file_priors": list(map(list, parser.file_priors.items())),
        "config_file_data": parser.config_file_data,
        "config_prefix": parser.config_prefix,
        "file_config_path": parser.file_config_path,
        "converter": None,
    }


def test_get_state_dict_after_parse_with_config_file(
    parser: OrionCmdlineParser, yaml_config: List[str], commandline: List[str]
):
    """Test getting state dict."""
    cmd_args = yaml_config
    cmd_args.extend(commandline)

    parser.parse(cmd_args)

    assert parser.get_state_dict() == {
        "parser": parser.parser.get_state_dict(),
        "cmd_priors": list(map(list, parser.cmd_priors.items())),
        "file_priors": list(map(list, parser.file_priors.items())),
        "config_file_data": parser.config_file_data,
        "config_prefix": parser.config_prefix,
        "file_config_path": parser.file_config_path,
        "converter": parser.converter.get_state_dict(),
    }


def test_set_state_dict(
    parser: OrionCmdlineParser,
    commandline: List[str],
    json_config: List[str],
    tmp_path: Path,
    json_converter,
    weird_argument: WeirdArgument,
):
    """Test that set_state_dict sets state properly to generate new config."""
    cmd_args = json_config
    cmd_args.extend(commandline)

    temp_parser: Optional[OrionCmdlineParser] = parser
    assert temp_parser is not None
    temp_parser.parse(cmd_args)

    state = temp_parser.get_state_dict()
    temp_parser = None

    blank_parser = OrionCmdlineParser(allow_non_existing_files=True)
    blank_parser.set_state_dict(state)

    trial = Trial(
        params=[
            {"name": "/lr", "type": "real", "value": -2.4},
            {"name": "/prior", "type": "categorical", "value": "sgd"},
            {"name": "/layers/1/width", "type": "integer", "value": 100},
            {"name": "/layers/1/type", "type": "categorical", "value": "relu"},
            {"name": "/layers/2/type", "type": "categorical", "value": "sigmoid"},
            {"name": "/training/lr0", "type": "real", "value": 0.032},
            {"name": "/training/mbs", "type": "integer", "value": 64},
            {"name": "/something-same", "type": "categorical", "value": "3"},
            {"name": "/a.b", "type": "real", "value": 0.2},
            {
                "name": f"/{weird_argument.name}",
                "type": weird_argument.prior_type,
                "value": weird_argument.value,
            },
        ]
    )

    output_file = str(tmp_path / "output.json")

    cmd_inst = blank_parser.format(output_file, trial)

    assert cmd_inst == [
        "--config",
        output_file,
        "--seed",
        "555",
        "--lr",
        "-2.4",
        "--non-prior",
        "choices({'sgd': 0.2, 'adam': 0.8})",
        "--prior",
        "sgd",
        "--a.b",
        "0.2",
        f"--{weird_argument.name}",
        f"{weird_argument.value}",
    ]

    output_data = json_converter.parse(output_file)
    assert output_data == {
        "yo": 5,
        "training": {"lr0": 0.032, "mbs": 64},
        "layers": [
            {"width": 64, "type": "relu"},
            {"width": 100, "type": "relu"},
            {"width": 16, "type": "sigmoid"},
        ],
        "something-same": "3",
    }
