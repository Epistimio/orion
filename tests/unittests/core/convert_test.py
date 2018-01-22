#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`metaopt.core.io.converter`."""

import pytest

from metaopt.core.io.convert import (infer_converter_from_file_type,
                                     JSONConverter, YAMLConverter)


def test_infer_a_yaml_converter(yaml_sample_path):
    """Infer a YAMLConverter from file path string."""
    yaml = infer_converter_from_file_type(yaml_sample_path)
    assert isinstance(yaml, YAMLConverter)


def test_infer_a_json_converter(json_sample_path):
    """Infer a JSONConverter from file path string."""
    json = infer_converter_from_file_type(json_sample_path)
    assert isinstance(json, JSONConverter)


def test_not_supported_file_ext():
    """Check what happens if invalid ext is given."""
    with pytest.raises(NotImplementedError) as exc:
        infer_converter_from_file_type("../naedw.py")
    assert "Supported" in str(exc.value)
