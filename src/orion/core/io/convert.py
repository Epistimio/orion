# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.convert` -- Parse and generate user script's configuration
====================================================================================

.. module:: convert
   :platform: Unix
   :synopsis: Defines and instantiates a converter for configuration file types.

Given a file path infer which configuration file parser/emitter it corresponds to.
Define `Converter` classes with a common interface for many popular configuration
file types.

Currently supported:
    - YAML
    - JSON

"""
from abc import (ABC, abstractmethod)
import importlib
import os

from orion.core.utils import Factory


def infer_converter_from_file_type(config_path):
    """Use filetype extension to infer and build the correct configuration file
    converter.
    """
    _, ext_type = os.path.splitext(os.path.abspath(config_path))
    for klass in Converter.types:
        if ext_type in klass.file_extensions:
            return klass()

    error = "Could not recognise file extension '{0}'.".format(ext_type)
    error += "Supported converters: {0}".format(Converter.types)
    raise NotImplementedError(error)


class BaseConverter(ABC):
    """Base class for configuration parsers/generators.

    Attributes
    ----------
    file_extensions : list of strings
       Strings starting with '.' which identify usually a file type as a
       common convention. For instance, ``['.yml', '.yaml']`` for YAML files.

    """

    file_extensions = []

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

    file_extensions = ['.yml', '.yaml']

    def __init__(self):
        """Try to dynamically import yaml module."""
        self.yaml = importlib.import_module('yaml')

    def parse(self, filepath):
        """Read dictionary out of the configuration file.

        Parameters
        ----------
        file : str
           Full path to the original user script's configuration.

        """
        with open(filepath) as f:
            return self.yaml.load(stream=f)

    def generate(self, filepath, data):
        """Create a configuration file at `filepath` using dictionary `data`."""
        with open(filepath, 'w') as f:
            self.yaml.dump(data, stream=f)


class JSONConverter(BaseConverter):
    """Converter for JSON files."""

    file_extensions = ['.json']

    def __init__(self):
        """Try to dynamically import json module."""
        self.json = importlib.import_module('json')

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
        with open(filepath, 'w') as f:
            self.json.dump(data, f)


# pylint: disable=too-few-public-methods,abstract-method
class Converter(BaseConverter, metaclass=Factory):
    """Class used to inject dependency on a configuration file parser/generator.

    .. seealso:: `Factory` metaclass and `BaseConverter` interface.
    """

    pass
