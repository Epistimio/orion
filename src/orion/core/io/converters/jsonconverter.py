import importlib

from orion.core.io.converters.base import BaseConverter


class JSONConverter(BaseConverter):
    """Converter for JSON files."""

    file_extensions = ['.json']

    def __init__(self, **kwargs):
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
