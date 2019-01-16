#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.orion_cmdline_parser`."""
import pytest

from collections import OrderedDict
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser


@pytest.fixture
def no_priors():
    """Return a simple commandline arguments list."""
    return ['1', '-a', '2', '--bbb', '3']


@pytest.fixture
def with_config(no_priors):
    """Return a simple commandline arguments list augmented with a config file."""
    return no_priors.append(['--config', 'orion.yaml'])


@pytest.fixture
def with_different_config(no_priors):
    """Return a simple commandline arguments list augmented with a different prefix for the
    config file"""
    return no_priors.append(['--other', 'orion.yaml'])


@pytest.fixture
def with_priors(no_priors):
    """Return a simple commandline arguments list augmented with priors."""
    return no_priors.append(['--cc~normal(0,1)', '--d~uniform(0,10)', '-f', 'orion~normal(0,1)'])


@pytest.fixture
def with_priors_replaced():
    """Return a simple commandline arguments list with expected replacement for priors."""
    return ['1', '-a', '2', '--bbb', '3', '--config', 'orion.yaml', '--cc', 'orion~normal(0,1)',
            '-d', 'orion~uniform(0,10)', '-f', 'orion~normal(0,1)']


@pytest.fixture
def with_templates(no_priors):
    """Return a simple commandline arguments list augmented with templates."""
    return no_priors.append(['{trial.hash_name}', '-g', '{trial.hash_name}'])


@pytest.fixture
def valid_priors_list():
    """Return a dictionary of valid and invalid priors for pattern matching."""
        return OrderedDict('x~normal(0,1)', 'x~+normal(0,1)', 'x~-normal(0,1)',
            'orion~>x', 'orion~loguniform(1e-5, 1, shape=10)', 'orion~>{trial.id}')


@pytest.fixture
def parser():
    """Return a defaulted `OrionCmddlineParser`."""
    return OrionCmdlineParser()


@pytest.fixture
def parser_diff_prefix():
    """Return a `OrionCmdlineParser` with a different config prefix."""
    return OrionCmdlineParser(config_prefix='--other')


def test_normal_parsing(no_priors, parser):
    """Parse a totally normal commandline arguments list."""
    parser.parse(no_priors)

    assert parser.parser.arguments == no_priors


def test_find_config_file(with_config, parser):
    """Check if the config file has been found through the command line."""
    parser.parse(with_config)

    assert parser.file_config_path == 'orion.yaml'
    # TODO: Add assertion for content of config file


def test_different_prefix(with_different_config, parser_diff_prefix):
    """Check if the config file has been found through the command line with another prefix."""
    parser_diff_prefix.parse(with_different_config)

    assert parser.file_config_path == 'orion.yaml'


def test_replace_prior(with_priors, with_priors_replaced):
    """Test the `_replace_prior` function to check if every ways to describe
    a parameter is being handled.
    """
    cmdline_parser = OrionCmdlineParser()

    replaced = cmdline_parser._replace_priors(with_priors)

    assert replaced == with_priors_replaced


def test_extract_prior(parser, valid_priors_list):
    """Check if the regex for extracting priors work as well as making sure
    they are correctly formatted"""
    for prior in valid_priors_list:
        parser._extract_prior(prior, extracted)
        assert extracted.keys()[0] == 
