#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.config`."""

import os

import pytest
import yaml

from orion.core.io.config import Configuration, ConfigurationError


@pytest.fixture
def yaml_path():
    """Create a temporary yaml file and return the path"""
    file_path = './my.yaml'
    with open(file_path, 'w') as f:
        f.write(yaml.dump({'test': 'from my yaml!'}))

    yield file_path

    os.remove(file_path)


def test_fetch_non_existing_option():
    """Test that access to a non existing key raises ConfigurationError"""
    config = Configuration()
    with pytest.raises(ConfigurationError) as exc:
        config.voici_voila

    assert 'Configuration does not have an attribute \'voici_voila\'.' in str(exc.value)


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
    config.add_option('test', option_type=int)
    with pytest.raises(ConfigurationError) as exc:
        config.test

    assert 'Configuration not set and no default provided: test.' in str(exc.value)


def test_set_non_existing_option():
    """Test that setting a non existing option crash"""
    config = Configuration()
    with pytest.raises(TypeError) as exc:
        config.test = 1
    assert "Can only set test as a Configuration, not <class 'int'>" in str(exc.value)


def test_set_subconfig_over_option():
    """Test that overwritting an option with a subconfig is not possible"""
    config = Configuration()
    config.add_option('test', option_type=int)
    config.test = 1
    assert config.test == 1
    with pytest.raises(TypeError) as exc:
        config.test = Configuration()
    assert "Cannot overwrite option test with a configuration" in str(exc.value)


def test_set_int_value():
    """Test that an integer option can have its value set"""
    config = Configuration()
    config.add_option('test', option_type=int)

    with pytest.raises(ConfigurationError) as exc:
        config.test

    config.test = 1
    assert config.test == 1
    config.test = "1"
    assert config.test == 1
    with pytest.raises(TypeError) as exc:
        config.test = "voici_voila"
    assert "<class 'int'> cannot be set to voici_voila with type <class 'str'>" in str(exc.value)


def test_set_real_value():
    """Test that a float option can have its value set"""
    config = Configuration()
    config.add_option('test', option_type=float)

    with pytest.raises(ConfigurationError) as exc:
        config.test

    config.test = 1
    assert config.test == 1.0
    config.test = "1"
    assert config.test == 1.0
    with pytest.raises(TypeError) as exc:
        config.test = "voici_voila"
    assert "<class 'float'> cannot be set to voici_voila with type <class 'str'>" in str(exc.value)


def test_set_str_value():
    """Test that a string option can have its value set"""
    config = Configuration()
    config.add_option('test', option_type=str)

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
    config.sub.add_option('test', option_type=str)

    with pytest.raises(ConfigurationError):
        config.test

    config.sub.test = "1"
    assert config.sub.test == "1"
    config.sub.test = 1
    assert config.sub.test == "1"


def test_set_value_like_dict():
    """Test that we can set values like a dictionary"""
    config = Configuration()
    config.add_option('test', option_type=str)

    with pytest.raises(ConfigurationError):
        config.test

    config['test'] = "1"
    assert config.test == "1"
    config['test'] = 1
    assert config.test == "1"


def test_set_subconfig_value_like_dict():
    """Test that we can set values like a dictionary"""
    config = Configuration()
    config.sub = Configuration()
    config.sub.add_option('test', option_type=str)

    with pytest.raises(ConfigurationError):
        config.test

    config['sub.test'] = "1"
    assert config.sub.test == "1"
    config['sub.test'] = 1
    assert config.sub.test == "1"


def test_set_invalid_subconfig_value_like_dict():
    """Test that deep keys cannot be set if subconfig does not exist"""
    config = Configuration()
    with pytest.raises(BaseException) as exc:
        config['sub.test'] = "1"
    assert 'Configuration does not have an attribute \'sub\'.' in str(exc.value)


def test_yaml_loading_empty_config(yaml_path):
    """Test loading for empty config fails like setting non existing attributes."""
    config = Configuration()

    with pytest.raises(ConfigurationError) as exc:
        config.load_yaml(yaml_path)

    assert 'Configuration does not have an attribute \'test\'.' in str(exc.value)


def test_default_value():
    """Test that default value is given only when nothing else is available"""
    config = Configuration()
    config.add_option('test', option_type=str, default="voici_voila")
    assert config.test == "voici_voila"


def test_yaml_precedence(yaml_path):
    """Test that yaml definition has precedence over default values"""
    config = Configuration()
    config.add_option('test', option_type=str, default="voici_voila", env_var="TOP_SECRET_MESSAGE")
    assert config.test == "voici_voila"

    config.load_yaml(yaml_path)
    assert config.test == "from my yaml!"


def test_env_var_precedence(yaml_path):
    """Test that env_var has precedence over yaml values"""
    config = Configuration()
    config.add_option('test', option_type=str, default="voici_voila", env_var="TOP_SECRET_MESSAGE")
    assert config.test == "voici_voila"

    config.load_yaml(yaml_path)
    assert config.test == "from my yaml!"

    os.environ['TOP_SECRET_MESSAGE'] = 'coussi_coussa'
    assert config.test == "coussi_coussa"
    del os.environ['TOP_SECRET_MESSAGE']

    assert config.test == "from my yaml!"


def test_local_precedence(yaml_path):
    """Test local setting has precedence over env var values"""
    config = Configuration()
    config.add_option('test', option_type=str, default="voici_voila", env_var="TOP_SECRET_MESSAGE")
    assert config.test == "voici_voila"

    config.load_yaml(yaml_path)
    assert config.test == "from my yaml!"

    os.environ['TOP_SECRET_MESSAGE'] = 'coussi_coussa'
    assert config.test == "coussi_coussa"

    config.test = "comme_ci_comme_ca"
    assert config.test == "comme_ci_comme_ca"

    del os.environ['TOP_SECRET_MESSAGE']
