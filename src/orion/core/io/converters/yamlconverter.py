import importlib

from orion.core.io.converters.base import BaseConverter


class YAMLConverter(BaseConverter):
    """Converter for YAML files."""

    file_extensions = ['.yml', '.yaml']

    def __init__(self, **kwargs):
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
