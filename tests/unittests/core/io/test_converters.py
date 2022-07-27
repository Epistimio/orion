#!/usr/bin/env python
"""Example usage and tests for :mod:`orion.core.io.converter`."""
import os

import pytest

from orion.core.io.convert import (
    GenericConverter,
    JSONConverter,
    YAMLConverter,
    infer_converter_from_file_type,
)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_infer_a_yaml_converter(yaml_sample_path):
    """Infer a YAMLConverter from file path string."""
    yaml = infer_converter_from_file_type(yaml_sample_path)
    assert isinstance(yaml, YAMLConverter)


def test_infer_a_json_converter(json_sample_path):
    """Infer a JSONConverter from file path string."""
    json = infer_converter_from_file_type(json_sample_path)
    assert isinstance(json, JSONConverter)


def test_unknown_file_ext(unknown_type_sample_path):
    """Check what happens if an unknown config type is given."""
    converter = infer_converter_from_file_type(unknown_type_sample_path)
    assert isinstance(converter, GenericConverter)


@pytest.fixture()
def generic_converter():
    """Return a generic converter."""
    return GenericConverter(expression_prefix="o~")


class TestGenericConverter:
    """Test functionality of generic converter of unknown configuration file type."""

    def test_parse(self, unknown_type_sample_path, generic_converter):
        """Test parsing method."""
        ret = generic_converter.parse(unknown_type_sample_path)
        assert ret == {
            "lalala": "o~uniform(1, 3, shape=(100, 3))",
            "lala": {
                "la": "o~uniform(1, 3, shape=(100, 3))",
                "l2a": "o~+gaussian(0, 0.1, shape=(100, 3))",
            },
            "": {
                "lala": {
                    "iela": "o~uniform(1, 3, shape=(100, 3))",
                    "": {
                        "iela": "o~uniform(1, 3, shape=(100, 3))",
                    },
                },
            },
            "aaalispera": "o~normal(3, 1)",
            "a": "o~normal(5, 3)",
            "b": "o~>a_serious_name",
            "a_serious_name": "o~-",
            "another_serious_name": "o~loguniform(0.001, 0.5)",
        }

        assert (
            generic_converter.template
            == """{lalala!s}

{/lala/la!s}
{//lala/iela!s}
{//lala//iela!s}

a:
   b:
    [asdfa~/Iamapath/dont_capture_me,
      ~/Iamapath/dont_capture_me]

yo:
           {aaalispera!s}

[naedw]
a_var={a!s}

~

[naekei:naedw]
other_var = ~
# Rename it who_names_their_variables_a_seriously
a_var = {b!s}

{{'oups':
# remove it
'{a_serious_name!s}',
'iela':'{another_serious_name!s}'
}}

{lala/l2a!s}
"""
        )

    def test_bad_parse_1(self, generic_converter):
        """Check that conflict is reported when duplicates are present."""
        BAD_UNKNOWN_SAMPLE_1 = os.path.join(TEST_DIR, "..", "bad_config1.txt")
        with pytest.raises(ValueError) as exc:
            generic_converter.parse(BAD_UNKNOWN_SAMPLE_1)
        assert "/lala/la" in str(exc.value)

    def test_bad_parse_2(self, generic_converter):
        """Check that conflict is reported even when more sneaky duplicate happens."""
        BAD_UNKNOWN_SAMPLE_2 = os.path.join(TEST_DIR, "..", "bad_config2.txt")
        with pytest.raises(ValueError) as exc:
            generic_converter.parse(BAD_UNKNOWN_SAMPLE_2)
        assert "lala/la" in str(exc.value)

    def test_bad_parse_3(self, generic_converter):
        """Check that conflict is reported if a namespace points is both and not final."""
        BAD_UNKNOWN_SAMPLE_3 = os.path.join(TEST_DIR, "..", "bad_config3.txt")
        with pytest.raises(ValueError) as exc:
            generic_converter.parse(BAD_UNKNOWN_SAMPLE_3)
        assert "lala" in str(exc.value)

    def test_bad_parse_4(self, generic_converter):
        """Check that conflict is reported if a namespace points is both and not final."""
        BAD_UNKNOWN_SAMPLE_4 = os.path.join(TEST_DIR, "..", "bad_config4.txt")
        with pytest.raises(ValueError) as exc:
            generic_converter.parse(BAD_UNKNOWN_SAMPLE_4)
        assert "lala" in str(exc.value)

    def test_generate(self, tmpdir, generic_converter, unknown_type_sample_path):
        """Check generation of a particular instance of configuration."""
        generic_converter.parse(unknown_type_sample_path)

        output_file = str(tmpdir.join("output.lalal"))
        data = {
            "lalala": "ispi",
            "lala": {
                "la": 5,
                "l2a": "+gaussian(0, 0.1, shape=(100, 3))",
            },
            "": {
                "lala": {
                    "iela": 1,
                    "": {
                        "iela": 2,
                    },
                },
            },
            "aaalispera": 3,
            "a": 4,
            "b": "o~>a_serious_name",
            "a_serious_name": "o~-",
            "another_serious_name": [1, 3],
        }
        generic_converter.generate(output_file, data)

        with open(output_file) as f:
            out = f.read()
        assert (
            out
            == """ispi

5
1
2

a:
   b:
    [asdfa~/Iamapath/dont_capture_me,
      ~/Iamapath/dont_capture_me]

yo:
           3

[naedw]
a_var=4

~

[naekei:naedw]
other_var = ~
# Rename it who_names_their_variables_a_seriously
a_var = o~>a_serious_name

{'oups':
# remove it
'o~-',
'iela':'[1, 3]'
}

+gaussian(0, 0.1, shape=(100, 3))
"""
        )

    def test_get_state_dict(
        self, unknown_type_sample_path, unknown_type_template_path, generic_converter
    ):
        """Test getting state dict."""
        generic_converter.parse(unknown_type_sample_path)
        assert generic_converter.get_state_dict() == {
            "expression_prefix": "o~",
            "has_leading": {"/lala//iela": "/", "/lala/iela": "/", "lala/la": "/"},
            "regex": "([\\/]?[\\w|\\/|-]+)~([\\+]?.*\\)|\\-|\\>[A-Za-z_]\\w*)",
            "template": open(unknown_type_template_path).read(),
        }

    def test_set_state_dict(self, tmpdir, generic_converter):
        """Test that set_state_dict sets state properly to generate new config."""
        generic_converter.set_state_dict(
            {
                "expression_prefix": "",
                "has_leading": {"voici": "/"},
                "regex": generic_converter.regex.pattern,
                "template": """\
voici = {/voici}
voila = voici + {voila}""",
            }
        )

        output_file = str(tmpdir.join("output.lalal"))

        generic_converter.generate(
            output_file, {"/voici": "me voici", "voila": "me voila"}
        )

        with open(output_file) as f:
            out = f.read()

        assert (
            out
            == """\
voici = me voici
voila = voici + me voila"""
        )
