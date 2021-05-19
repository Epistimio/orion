# -*- coding: utf-8 -*-
"""
Parse and generate user script's configuration
==============================================

Defines and instantiates a converter for configuration file types.

Given a file path infer which configuration file parser/emitter it corresponds to.
Define `Converter` classes with a common interface for many popular configuration
file types.

Currently supported:
    - YAML
    - JSON
    - See below, for configuration agnostic parsing

A `GenericConverter` is provided that tries and parses configuration
files, regardless of their type, according to predefined Oríon's markers.

"""
import importlib
import os
from abc import ABC, abstractmethod
from collections import deque

from orion.core.utils import Factory, nesteddict


def infer_converter_from_file_type(config_path, regex=None, default_keyword=""):
    """Use filetype extension to infer and build the correct configuration file
    converter.
    """
    _, ext_type = os.path.splitext(os.path.abspath(config_path))
    for klass in Converter.types.values():
        if ext_type in klass.file_extensions:
            return klass()

    if regex is None:
        return GenericConverter(expression_prefix=default_keyword)

    return GenericConverter(regex, expression_prefix=default_keyword)


class BaseConverter(ABC):
    """Base class for configuration parsers/generators.

    Attributes
    ----------
    file_extensions : list of strings
       Strings starting with '.' which identify usually a file type as a
       common convention. For instance, ``['.yml', '.yaml']`` for YAML files.

    """

    file_extensions = []

    # pylint:disable=no-self-use
    def get_state_dict(self):
        """Give state dict that can be used to reconstruct the converter"""
        return {}

    def set_state_dict(self, state):
        """Reset the converter based on previous state"""
        pass

    @abstractmethod
    def parse(self, filepath):
        """Read dictionary out of the configuration file.

        Parameters
        ----------
        filepath : str
           Full path to the original user script's configuration.

        """
        pass

    @abstractmethod
    def generate(self, filepath, data):
        """Create a configuration file at `filepath` using dictionary `data`."""
        pass


class YAMLConverter(BaseConverter):
    """Converter for YAML files."""

    file_extensions = [".yml", ".yaml"]

    def __init__(self):
        """Try to dynamically import yaml module."""
        self.yaml = importlib.import_module("yaml")

    def parse(self, filepath):
        """Read dictionary out of the configuration file.

        Parameters
        ----------
        file : str
           Full path to the original user script's configuration.

        """
        with open(filepath) as f:
            return self.yaml.safe_load(stream=f)

    def generate(self, filepath, data):
        """Create a configuration file at `filepath` using dictionary `data`."""
        with open(filepath, "w") as f:
            self.yaml.dump(data, stream=f)


class JSONConverter(BaseConverter):
    """Converter for JSON files."""

    file_extensions = [".json"]

    def __init__(self):
        """Try to dynamically import json module."""
        self.json = importlib.import_module("json")

    def parse(self, filepath):
        """Read dictionary out of the configuration file.

        Parameters
        ----------
        file : str
           Full path to the original user script's configuration.

        """
        with open(filepath) as f:
            return self.json.load(f)

    def generate(self, filepath, data):
        """Create a configuration file at `filepath` using dictionary `data`."""
        with open(filepath, "w") as f:
            self.json.dump(data, f)


class GenericConverter(BaseConverter):
    """Generic converter for any configuration file type.

    For each parameter dimension declared here, one must necessarily
    provide a ``name`` keyword inside the `Dimension` building expression.

    Implementation details: As this class is supposed to provide with a
    generic text parser, semantics are going to be tied to their consequent
    usage. A template document is going to be created on `parse` and filled
    with values on `read`. This template document consists the state of this
    `Converter` object.

    Dimension should be defined for instance as:
    ``meaningful_name~uniform(0, 4)``

    """

    def __init__(
        self,
        regex=r"([\/]?[\w|\/|-]+)~([\+]?.*\)|\-|\>[A-Za-z_]\w*)",
        expression_prefix="",
    ):
        """Initialize with the regex expression which will be searched for
        to define a `Dimension`.
        """
        self.re_module = importlib.import_module("re")
        self.regex = self.re_module.compile(regex)
        self.expression_prefix = expression_prefix
        self.template = None
        self.has_leading = dict()
        self.conflict_msg = "Namespace conflict in configuration file '{}', under '{}'"

    def get_state_dict(self):
        """Give state dict that can be used to reconstruct the converter"""
        return dict(
            regex=self.regex.pattern,
            expression_prefix=self.expression_prefix,
            template=self.template,
            has_leading=self.has_leading,
        )

    def set_state_dict(self, state):
        """Reset the converter based on previous state"""
        self.regex = self.re_module.compile(state["regex"])
        self.expression_prefix = state["expression_prefix"]
        self.template = state["template"]
        self.has_leading = state["has_leading"]

    def _raise_conflict(self, path, namespace):
        raise ValueError(self.conflict_msg.format(path, namespace))

    def parse(self, filepath):
        r"""Read dictionary out of the configuration file.

        Create a template for Python 3 string format and save it as this
        object's state, by substituing '{\1}' wherever the pattern
        was matched. By default, the first matched group (\1) corresponds
        with a dimension's namespace.

        .. note:: Namespace in substitution templates does not contain the first '/'.

        Parameters
        ----------
        filepath : str
           Full path to the original user script's configuration.

        """
        with open(filepath) as f:
            self.template = f.read()

        # Search for Oríon semantic pattern
        pairs = self.regex.findall(self.template)
        ret = dict(pairs)

        # Every namespace given should be unique,
        # raise conflict if there are duplicates
        if len(pairs) != len(ret):
            namespaces = list(zip(*pairs))[0]
            for name in namespaces:
                if namespaces.count(name) != 1:
                    self._raise_conflict(filepath, name)

        # Create template using each namespace as format key,
        # exactly as provided by the user
        subst = self.re_module.sub(r"{", r"{{", self.template)
        subst = self.re_module.sub(r"}", r"}}", subst)
        substituted, num_subs = self.regex.subn(r"{\1!s}", subst)
        assert len(ret) == num_subs, (
            "This means an error in the regex. Report bug. Details::\n"
            "original: {}\n, regex:{}".format(self.template, self.regex)
        )
        self.template = substituted

        # Wrap it in style of what the rest of `Converter`s return
        ret_nested = nesteddict()
        for namespace, expression in ret.items():
            keys = namespace.split("/")
            if not keys[0]:  # It means that user wrote a namespace starting from '/'
                keys = keys[1:]  # Safe because of the regex pattern
                self.has_leading[namespace[1:]] = "/"

            stuff = ret_nested
            for i, key in enumerate(keys[:-1]):
                stuff = stuff[key]
                if isinstance(stuff, str):
                    # If `stuff` is not a dictionary while traversing the
                    # namespace path, then this amounts to a conflict which was
                    # not sufficiently get caught
                    self._raise_conflict(filepath, "/".join(keys[: i + 1]))
            # If final value is already filled,
            # then this must be also due to a conflict
            if stuff[keys[-1]]:
                self._raise_conflict(filepath, namespace)

            # Keep compatibility with `SpaceBuilder._build_from_config`
            stuff[keys[-1]] = self.expression_prefix + expression

        return ret_nested

    def generate(self, filepath, data):
        """Create a configuration file at `filepath` using dictionary `data`."""
        unnested_data = dict()
        stack = deque()
        stack.append(([], data))
        while True:
            try:
                namespace, stuff = stack.pop()
            except IndexError:
                break
            if isinstance(stuff, dict):
                for k, v in stuff.items():
                    stack.append((["/".join(namespace + [str(k)])], v))
            else:
                name = namespace[0]
                unnested_data[self.has_leading.get(name, "") + name] = stuff

        print(self.template)
        print(unnested_data)
        document = self.template.format(**unnested_data)

        with open(filepath, "w") as f:
            f.write(document)


# pylint: disable=too-few-public-methods,abstract-method
class Converter(BaseConverter, metaclass=Factory):
    """Class used to inject dependency on a configuration file parser/generator.

    .. seealso:: :class:`orion.core.utils.Factory` metaclass and `BaseConverter` interface.
    """

    pass
