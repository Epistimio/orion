# -*- coding: utf-8 -*-
"""
:mod:`orion.core.worker.convert` -- Parse and generate user script's configuration
==================================================================================

.. module:: convert
   :platform: Unix
   :synopsis: Defines and instantiates a converter for configuration file types.

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
from abc import (ABCMeta, abstractmethod)
import os

from orion.core.utils import (Concept, Wrapper)


class BaseConverter(Concept, metaclass=ABCMeta):
    """Base class for configuration parsers/generators.

    Attributes
    ----------
    file_extensions : list of strings
       Strings starting with '.' which identify usually a file type as a
       common convention. For instance, ``['.yml', '.yaml']`` for YAML files.

    """

    name = "Converter"

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


# pylint: disable=too-few-public-methods,abstract-method
class Converter(Wrapper):

    implementation_module = "orion.core.io.converters"

    file_extensions = {'.yml': 'yaml', '.yaml': 'yaml', '.json': 'json'}

    def __init__(self, config_path, regex=None, default_keyword=''):
        _, ext_type = os.path.splitext(os.path.abspath(config_path))

        classname = self.file_extensions.get(ext_type, 'generic') + 'converter'

        print(regex)
        print(default_keyword)
        instance_dict = {classname: dict(regex=regex, expression_prefix=default_keyword)}
        super(Converter, self).__init__(instance=instance_dict)

    @property
    def wraps(self):
        return BaseConverter
