#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.core.io.config`."""

import argparse
import logging
import os

import pytest
import yaml

from orion.core.io.config import Configuration, ConfigurationError


@pytest.fixture
def yaml_path():
    """Create a temporary yaml file and return the path"""
    file_path = "./my.yaml"
    with open(file_path, "w") as f:
        f.write(yaml.dump({"test": "from my yaml!"}))

    yield file_path

    os.remove(file_path)


@pytest.fixture
def broken_yaml_path():
    """Create a temporary yaml file and return the path"""
    file_path = "./my.yaml"
    with open(file_path, "w") as f:
        f.write(yaml.dump({"coucou": "from my yaml!"}))

    yield file_path

    os.remove(file_path)


@pytest.fixture
def subdict_yaml_path():
    """Create a temporary yaml file with subdicts and return the path"""
    file_path = "./my.yaml"
    with open(file_path, "w") as f:
        f.write(
            yaml.dump(
                {
                    "test": {"i am": "a sub-dict"},
                    "test2": {"test": {"me is": "sub-conf sub-dict"}},
                }
            )
        )

    yield file_path

    os.remove(file_path)


def test_fetch_non_existing_option():
    """Test that access to a non existing key raises ConfigurationError"""
    config = Configuration()
    with pytest.raises(ConfigurationError) as exc:
        config.voici_voila

    assert "Configuration does not have an attribute 'voici_voila'." in str(exc.value)


def test_access_to_config():
    """Test that access to _config returns properly _config.

    This is because getattr() could grasp any key including `_config` and makes
    it impossible to access the later
    """
    assert Configuration()._config == {}


def test_set_subconfig():
    """Test that setting a subconfig works"""
    config = Configuration()
    config.test = Configuration()

    assert isinstance(config.test, Configuration)

    with pytest.raises(ConfigurationError):
        config.test.voici_voila


def test_access_config_without_values():
    """Test that access to config without values raises ConfigurationError"""
    config = Configuration()
    config.add_option("test", option_type=int)
    with pytest.raises(ConfigurationError) as exc:
        config.test

    assert "Configuration not set and no default provided: test." in str(exc.value)


def test_set_non_existing_option():
    """Test that setting a non existing option crash"""
    config = Configuration()
    with pytest.raises(TypeError) as exc:
        config.test = 1
    assert "Can only set test as a Configuration, not <class 'int'>" in str(exc.value)


def test_set_subconfig_over_option():
    """Test that overwriting an option with a subconfig is not possible"""
    config = Configuration()
    config.add_option("test", option_type=int)
    config.test = 1
    assert config.test == 1
    with pytest.raises(TypeError) as exc:
        config.test = Configuration()
    assert "Cannot overwrite option test with a configuration" in str(exc.value)


def test_set_int_value():
    """Test that an integer option can have its value set"""
    config = Configuration()
    config.add_option("test", option_type=int)

    with pytest.raises(ConfigurationError) as exc:
        config.test

    config.test = 1
    assert config.test == 1
    config.test = "1"
    assert config.test == 1
    with pytest.raises(TypeError) as exc:
        config.test = "voici_voila"
    assert "<class 'int'> cannot be set to voici_voila with type <class 'str'>" in str(
        exc.value
    )


def test_set_real_value():
    """Test that a float option can have its value set"""
    config = Configuration()
    config.add_option("test", option_type=float)

    with pytest.raises(ConfigurationError) as exc:
        config.test

    config.test = 1
    assert config.test == 1.0
    config.test = "1"
    assert config.test == 1.0
    with pytest.raises(TypeError) as exc:
        config.test = "voici_voila"
    assert (
        "<class 'float'> cannot be set to voici_voila with type <class 'str'>"
        in str(exc.value)
    )


def test_set_str_value():
    """Test that a string option can have its value set"""
    config = Configuration()
    config.add_option("test", option_type=str)

    with pytest.raises(ConfigurationError):
        config.test

    config.test = "1"
    assert config.test == "1"
    config.test = 1
    assert config.test == "1"


def test_set_value_of_subconfig_directly():
    """Test that we can access subconfig and set value directly"""
    config = Configuration()
    config.sub = Configuration()
    config.sub.add_option("test", option_type=str)

    with pytest.raises(ConfigurationError):
        config.test

    config.sub.test = "1"
    assert config.sub.test == "1"
    config.sub.test = 1
    assert config.sub.test == "1"


def test_set_value_like_dict():
    """Test that we can set values like a dictionary"""
    config = Configuration()
    config.add_option("test", option_type=str)

    with pytest.raises(ConfigurationError):
        config.test

    config["test"] = "1"
    assert config.test == "1"
    config["test"] = 1
    assert config.test == "1"


def test_set_subconfig_value_like_dict():
    """Test that we can set values like a dictionary"""
    config = Configuration()
    config.sub = Configuration()
    config.sub.add_option("test", option_type=str)

    with pytest.raises(ConfigurationError):
        config.test

    config["sub.test"] = "1"
    assert config.sub.test == "1"
    config["sub.test"] = 1
    assert config.sub.test == "1"


def test_set_invalid_subconfig_value_like_dict():
    """Test that deep keys cannot be set if subconfig does not exist"""
    config = Configuration()
    with pytest.raises(BaseException) as exc:
        config["sub.test"] = "1"
    assert "Configuration does not have an attribute 'sub'." in str(exc.value)


def test_yaml_loading_empty_config(yaml_path):
    """Test loading for empty config fails like setting non existing attributes."""
    config = Configuration()

    with pytest.raises(ConfigurationError) as exc:
        config.load_yaml(yaml_path)

    assert "Configuration does not have an attribute 'test'." in str(exc.value)


def test_default_value():
    """Test that default value is given only when nothing else is available"""
    config = Configuration()
    config.add_option("test", option_type=str, default="voici_voila")
    assert config.test == "voici_voila"


def test_yaml_precedence(yaml_path):
    """Test that yaml definition has precedence over default values"""
    config = Configuration()
    config.add_option(
        "test", option_type=str, default="voici_voila", env_var="TOP_SECRET_MESSAGE"
    )
    assert config.test == "voici_voila"

    config.load_yaml(yaml_path)
    assert config.test == "from my yaml!"


def test_env_var_precedence(yaml_path):
    """Test that env_var has precedence over yaml values"""
    config = Configuration()
    config.add_option(
        "test", option_type=str, default="voici_voila", env_var="TOP_SECRET_MESSAGE"
    )
    assert config.test == "voici_voila"

    config.load_yaml(yaml_path)
    assert config.test == "from my yaml!"

    os.environ["TOP_SECRET_MESSAGE"] = "coussi_coussa"
    assert config.test == "coussi_coussa"
    del os.environ["TOP_SECRET_MESSAGE"]

    assert config.test == "from my yaml!"


def test_env_var_list(yaml_path):
    """Test that env_var lists are correctly handled"""
    config = Configuration()
    config.add_option(
        "test", option_type=list, default=["voici"], env_var="TOP_SECRET_LIST"
    )
    assert config.test == ["voici"]

    os.environ["TOP_SECRET_LIST"] = "voila:voici:voila"
    assert config.test == ["voila", "voici", "voila"]


def test_local_precedence(yaml_path):
    """Test local setting has precedence over env var values"""
    config = Configuration()
    config.add_option(
        "test", option_type=str, default="voici_voila", env_var="TOP_SECRET_MESSAGE"
    )
    assert config.test == "voici_voila"

    config.load_yaml(yaml_path)
    assert config.test == "from my yaml!"

    os.environ["TOP_SECRET_MESSAGE"] = "coussi_coussa"
    assert config.test == "coussi_coussa"

    config.test = "comme_ci_comme_ca"
    assert config.test == "comme_ci_comme_ca"

    del os.environ["TOP_SECRET_MESSAGE"]


def test_overwrite_subconfig():
    """Test that subconfig cannot be overwritten"""
    config = Configuration()
    config.nested = Configuration()
    with pytest.raises(ValueError) as exc:
        config.add_option("nested", option_type=str)
    assert "Configuration already contains nested" == str(exc.value)

    with pytest.raises(ValueError) as exc:
        config.nested = Configuration()
    assert "Configuration already contains subconfiguration nested" == str(exc.value)


def test_load_yaml_with_dict_items(subdict_yaml_path):
    """Test that yaml config with items assigned with dicts is supported"""
    config = Configuration()
    default = {"default": "sub-dict"}
    config.add_option("test", option_type=dict, default=default)
    assert config.test == default
    config.test2 = Configuration()
    config.test2.add_option("test", option_type=dict, default=default)
    assert config.test2.test == default

    config.load_yaml(subdict_yaml_path)
    assert config.test == {"i am": "a sub-dict"}
    assert config.test2.test == {"me is": "sub-conf sub-dict"}


def test_load_yaml_unknown_option(broken_yaml_path):
    """Test error message when yaml config contains unknown options"""
    config = Configuration()
    config.add_option("test", option_type=str, default="hello")
    assert config.test == "hello"

    with pytest.raises(ConfigurationError) as exc:
        config.load_yaml(broken_yaml_path)
    assert exc.match("Configuration does not have an attribute 'coucou'")


def test_to_dict():
    """Test dictionary representation of the configuration"""
    config = Configuration()
    config.add_option("test", option_type=str, default="voici_voila")
    config.nested = Configuration()
    config.nested.add_option("test2", option_type=str, default="zici")

    assert config.to_dict() == {"test": "voici_voila", "nested": {"test2": "zici"}}

    config.test = "hello"
    config.nested.test2 = "labas"

    assert config.to_dict() == {"test": "hello", "nested": {"test2": "labas"}}


def test_key_curation():
    """Test that both - and _ maps to same options"""
    config = Configuration()
    config.add_option("test-with-dashes", option_type=int, default=1)
    config.add_option("test_with_underscores", option_type=int, default=2)
    config.add_option("test-all_mixedup", option_type=int, default=3)

    assert config["test-with-dashes"] == 1
    assert config["test_with_underscores"] == 2
    assert config["test-all_mixedup"] == 3

    assert config["test_with_dashes"] == 1
    assert config["test-with-underscores"] == 2
    assert config["test_all-mixedup"] == 3

    assert config.test_with_dashes == 1
    assert config.test_with_underscores == 2
    assert config.test_all_mixedup == 3

    config["test_with_dashes"] = 4
    assert config["test-with-dashes"] == 4


def test_nested_key_curation():
    """Test that both - and _ maps to same options in nested configs as well"""
    config = Configuration()
    config.add_option("test-with-dashes", option_type=str, default="voici_voila")
    config.nested = Configuration()
    config.nested.add_option("test_with_underscores", option_type=str, default="zici")

    assert config["nested"]["test_with_underscores"] == "zici"
    assert config["nested"]["test-with-underscores"] == "zici"

    config["nested"]["test-with-underscores"] = "labas"
    assert config.nested.test_with_underscores == "labas"


def test_help_option():
    """Verify adding documentation to options."""
    config = Configuration()
    config.add_option("option", option_type=str, help="A useless option!")

    assert config.help("option") == "A useless option!"


def test_help_nested_option():
    """Verify adding documentation to a nested option."""
    config = Configuration()
    config.add_option("option", option_type=str, help="A useless option!")
    config.nested = Configuration()
    config.nested.add_option("option", option_type=str, help="A useless nested option!")

    assert config.help("nested.option") == "A useless nested option!"
    assert config.nested.help("option") == "A useless nested option!"


def test_help_option_with_default():
    """Verify adding documentation to options with default value."""
    config = Configuration()
    config.add_option("option", option_type=str, default="a", help="A useless option!")

    assert config.help("option") == "A useless option! (default: a)"


def test_no_help_option():
    """Verify not adding documentation to options."""
    config = Configuration()
    config.add_option("option", option_type=str)

    assert config.help("option") == "Undocumented"


def test_argument_parser():
    """Verify the argument parser built based on config."""
    config = Configuration()
    config.add_option("option", option_type=str)

    parser = argparse.ArgumentParser()
    config.add_arguments(parser)

    options = parser.parse_args(["--option", "a"])

    assert options.option == "a"


def test_argument_parser_ignore_default():
    """Verify the argument parser does not get default values."""
    config = Configuration()
    config.add_option("option", option_type=str, default="b")

    parser = argparse.ArgumentParser()
    config.add_arguments(parser)

    options = parser.parse_args([])

    assert options.option is None


def test_argument_parser_rename():
    """Verify the argument parser built based on config with some options renamed."""
    config = Configuration()
    config.add_option("option", option_type=str)

    parser = argparse.ArgumentParser()
    config.add_arguments(parser, rename=dict(option="--new-option"))

    with pytest.raises(SystemExit) as exc:
        options = parser.parse_args(["--option", "a"])

    assert exc.match("2")

    options = parser.parse_args(["--new-option", "a"])

    assert options.new_option == "a"


def test_argument_parser_dict_list_tuple():
    """Verify the argument parser does not contain options of type dict/list/tuple in config."""
    config = Configuration()
    config.add_option("st", option_type=str)
    config.add_option("di", option_type=dict)
    config.add_option("li", option_type=list)
    config.add_option("tu", option_type=tuple)

    parser = argparse.ArgumentParser()
    config.add_arguments(parser)

    options = parser.parse_args([])
    assert vars(options) == {"st": None}

    with pytest.raises(SystemExit) as exc:
        options = parser.parse_args(["--di", "ct"])

    assert exc.match("2")


def test_deprecate_option(caplog):
    """Test deprecating an option."""
    config = Configuration()
    config.add_option(
        "option",
        option_type=str,
        default="hello",
        deprecate=dict(version="v1.0", alternative="None! T_T"),
    )

    config.add_option("ok", option_type=str, default="hillo")

    # Access the deprecated option and trigger a warning.
    with caplog.at_level(logging.WARNING, logger="orion.core.io.config"):
        assert config.option == "hello"

    assert caplog.record_tuples == [
        (
            "orion.core.io.config",
            logging.WARNING,
            "(DEPRECATED) Option `option` will be removed in v1.0. Use `None! T_T` instead.",
        )
    ]

    caplog.clear()

    # Access the non-deprecated option and trigger no warnings.
    with caplog.at_level(logging.WARNING, logger="orion.core.io.config"):
        assert config.ok == "hillo"

    assert caplog.record_tuples == []


def test_deprecate_option_missing_version():
    """Verify option deprecation if version is missing."""
    config = Configuration()
    with pytest.raises(ValueError) as exc:
        config.add_option(
            "option", option_type=str, deprecate=dict(alternative="None! T_T")
        )

    assert exc.match("`version` is missing in deprecate option")


def test_deprecate_option_no_alternative(caplog):
    """Verify option deprecation when there is no alternative."""
    config = Configuration()
    config.add_option(
        "option", option_type=str, default="hello", deprecate=dict(version="v1.0")
    )

    # Access the deprecated option and trigger a warning.
    with caplog.at_level(logging.WARNING, logger="orion.core.io.config"):
        assert config.option == "hello"

    assert caplog.record_tuples == [
        (
            "orion.core.io.config",
            logging.WARNING,
            "(DEPRECATED) Option `option` will be removed in v1.0.",
        )
    ]


def test_deprecate_option_help():
    """Verify help message of a deprecated option."""
    config = Configuration()
    config.add_option(
        "option",
        option_type=str,
        deprecate=dict(version="v1.0", alternative="None! T_T"),
        help="A useless option!",
    )

    assert config.help("option") == "(DEPRECATED) A useless option!"


def test_deprecate_option_print_with_different_name(caplog):
    """Verify deprecation warning with different name (for nested options)."""
    config = Configuration()
    config.add_option(
        "option",
        option_type=str,
        default="hello",
        deprecate=dict(version="v1.0", alternative="None! T_T", name="nested.option"),
    )

    # Access the deprecated option and trigger a warning.
    with caplog.at_level(logging.WARNING, logger="orion.core.io.config"):
        assert config.option == "hello"

    assert caplog.record_tuples == [
        (
            "orion.core.io.config",
            logging.WARNING,
            "(DEPRECATED) Option `nested.option` will be removed in v1.0. Use `None! T_T` instead.",
        )
    ]


def test_get_deprecated_key(caplog):
    """Verify deprecation warning using get()."""
    config = Configuration()
    config.add_option(
        "option",
        option_type=str,
        default="hello",
        deprecate=dict(version="v1.0", alternative="None! T_T"),
    )

    # Access the deprecated option and trigger a warning.
    with caplog.at_level(logging.WARNING, logger="orion.core.io.config"):
        assert config.get("option") == "hello"

    assert caplog.record_tuples == [
        (
            "orion.core.io.config",
            logging.WARNING,
            "(DEPRECATED) Option `option` will be removed in v1.0. Use `None! T_T` instead.",
        )
    ]


def test_get_deprecated_key_ignore_warning(caplog):
    """Verify deprecation warning using get(deprecated='ignore')."""
    config = Configuration()
    config.add_option(
        "option",
        option_type=str,
        default="hello",
        deprecate=dict(version="v1.0", alternative="None! T_T"),
    )

    # Access the deprecated option and trigger a warning.
    with caplog.at_level(logging.WARNING, logger="orion.core.io.config"):
        assert config.get("option", deprecated="ignore") == "hello"

    assert caplog.record_tuples == []


def test_from_dict_additional_keys_are_ignored():
    config = Configuration()
    config.from_dict(dict(undefined="are ignored"))
    assert config.to_dict() == dict()


def test_from_dict_simple_value():
    config = Configuration()
    config.add_option(
        "option",
        option_type=str,
        default="hello",
    )

    values = dict(option="123")
    config.from_dict(values)
    assert config.to_dict() == values


def test_from_dict_simple_value_value_error():
    config = Configuration()
    config.add_option(
        "option",
        option_type=float,
        default="hello",
    )

    values = dict(option="1232")
    with pytest.raises(ValueError):
        config.from_dict(values)


def test_from_dict_old_values_are_popped():
    config = Configuration()
    config.add_option(
        "option1",
        option_type=str,
        default="hello",
    )

    config.add_option(
        "option2",
        option_type=str,
        default="hello",
    )

    default = config.to_dict()

    # Override everything
    overrides = dict(option1="123", option2="123")
    config.from_dict(overrides)
    assert config.to_dict() == overrides

    #
    overrides.pop("option2")
    config.from_dict(overrides)

    # Get the expected value
    default.update(overrides)

    assert config.to_dict() == default


def test_from_dict_nested_config():
    config = Configuration()
    nested = Configuration()
    nested.add_option(
        "option1",
        option_type=str,
        default="hello",
    )
    config.sub = nested

    values = dict(sub=dict(option1="123"))
    config.from_dict(values)
    assert config.to_dict() == values


def test_from_dict_nested_values_are_popped():
    config = Configuration()
    nested = Configuration()
    nested.add_option(
        "option1",
        option_type=str,
        default="hello",
    )
    nested.add_option(
        "option2",
        option_type=str,
        default="hello",
    )
    config.sub = nested
    default = nested.to_dict()

    # Override everything
    overrides = dict(option1="123", option2="123")
    all = dict(sub=overrides)

    config.from_dict(all)
    assert config.to_dict() == all

    # Remove one override
    overrides.pop("option2")
    partial = dict(sub=overrides)

    # Get the expected result by applying overrides to the default config
    default.update(overrides)
    expected = dict(sub=default)

    config.from_dict(partial)
    assert config.to_dict() == expected
