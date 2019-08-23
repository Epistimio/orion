# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.config` -- Configuration object
===================================================

.. module:: config
   :platform: Unix
   :synopsis: Configuration object to define package configuration


Highly inspired from https://github.com/mila-iqia/blocks/blob/master/blocks/config.py.

"""
import logging
import os

import yaml

from orion.core.utils.flatten import flatten


logger = logging.getLogger(__name__)


NOT_SET = object()


class ConfigurationError(Exception):
    """Error raised when a configuration value is requested but not set."""


class Configuration:
    """Configuration object

    Provides default values configurable at different levels. The configuration object can have
    global default values, which may be overriden by user with yaml configuration files, environment
    variables or by setting directly the values in the configuration object. In order, direct
    definition overrides, environment variables, which overrides yaml configuration, which overrides
    default values in configuration object definition.

    Examples
    --------
    >>> config = Configuration()
    >>> config.add_option('test', str, 'hello', 'TEST_ENV')
    >>> config.test
    'hello'
    >>> config.load_yaml('some_config.yaml')
    >>> config.test
    'in yaml!'
    >>> os.environ['TEST_ENV'] = 'there!'
    >>> config.test
    'there!'
    >>> config.test = 'here'
    >>> config.test
    'here'

    """

    SPECIAL_KEYS = ['_config', '_yaml', '_default', '_env_var']

    def __init__(self):
        self._config = {}

    def load_yaml(self, path):
        """Load yaml file and set global default configuration

        Parameters
        ----------
        path: str
            Path to the global configuration file.

        Raises
        -------
        ConfigurationError
            If some option in the yaml file does not exist in the config

        """
        with open(path) as f:
            cfg = yaml.safe_load(f)
            if cfg is None:
                return
            # implies that yaml must be in dict form
            for key, value in flatten(cfg).items():
                default = self[key]
                logger.debug('Overwritting "%s" default %s with %s', key, default, value)
                self[key + '._yaml'] = value

    def __getattr__(self, key):
        """Get the value of the option

        Parameters
        ----------
        key: str
            Name of the option

        Returns
        -------
            Value of the option.

        Raises
        -------
        ConfigurationError
            If the option does not exist

        """
        if key == 'config':
            raise AttributeError
        if key not in self._config:
            raise ConfigurationError("Configuration does not have an attribute "
                                     "'{}'.".format(key))

        config_setting = self._config[key]
        if 'value' in config_setting:
            value = config_setting['value']
        elif ('env_var' in config_setting and
              config_setting['env_var'] in os.environ):
            value = os.environ[config_setting['env_var']]
        elif 'yaml' in config_setting:
            value = config_setting['yaml']
        elif 'default' in config_setting:
            value = config_setting['default']
        else:
            raise ConfigurationError("Configuration not set and no default "
                                     "provided: {}.".format(key))

        return config_setting['type'](value)

    def __setattr__(self, key, value):
        """Set option value or subconfiguration

        Parameters
        ----------
        key: str
            The key or namespace to set the value of the configuration.
            If the configuration has subconfiguration, the key may be
            hierarchical with each level seperated by dots.
            Ex: 'first.second.third'
        value: object or Configuration
            A general object to set an option or a configuration object to set
            a sub configuration.

        Raises
        ------
        TypeError
            - If value is a configuration and an option is already defined for
            given key, or
            - If the value has an invalid type for the given option, or
            - If no option exists for the given key and the value is not a
            configuration object.

        """
        if key not in self.SPECIAL_KEYS and key in self._config:
            self._validate(key, value)
            self._config[key]['value'] = value

        elif key == '_config' or isinstance(value, Configuration):
            super(Configuration, self).__setattr__(key, value)

        else:
            raise TypeError("Can only set {} as a Configuration, not {}. Use add_option to set a "
                            "new option.".format(key, type(value)))

    def _validate(self, key, value):
        """Validate the (key, value) option

        Raises
        ------
        TypeError
            - If the value is a Configuration object while the key is defined as a normal field
              (str, int, bool, etc). It cannot be overwritten by a subconfiguration.
            - If the type of `value` is not valid for `key`.

        """
        if isinstance(value, Configuration):
            raise TypeError("Cannot overwrite option {} with a configuration".format(key))

        try:
            self._config[key]['type'](value)
        except ValueError as e:
            message = "Option {} of type {} cannot be set to {} with type {}".format(
                key, self._config[key]['type'], value, type(value))
            raise TypeError(message) from e

    def __setitem__(self, key, value):
        """Set option value using dict-like syntax

        Parameters
        ----------
        key: str
            The key or namespace to set the value of the configuration.
            If the configuration has subconfiguration, the key may be
            hierarchical with each levels seperated by dots.
            Ex: 'first.second.third'
        value: object
            A general object to set an option.

        """
        keys = key.split(".")

        # Set in current config for special keys
        if len(keys) == 2 and keys[-1] in self.SPECIAL_KEYS:
            key, field = keys
            self._validate(key, value)
            self._config[key][field.lstrip('_')] = value

        # Set in current configuration
        elif len(keys) == 1:
            setattr(self, keys[0], value)

        # Recursively in sub configurations
        else:
            subconfig = getattr(self, keys[0])
            if subconfig is None:
                raise KeyError("'{}' is not defined in configuration.".format(keys[0]))
            subconfig[".".join(keys[1:])] = value

    def __getitem__(self, key):
        """Get option value using dict-like syntax

        Parameters
        ----------
        key: str
            The key or namespace to set the value of the configuration.
            If the configuration has subconfiguration, the key may be
            hierarchical with each levels seperated by dots.
            Ex: 'first.second.third'

        """
        keys = key.split(".")
        # Recursively in sub configurations
        if len(keys) > 1:
            subconfig = getattr(self, keys[0])
            if subconfig is None:
                raise KeyError("'{}' is not defined in configuration.".format(keys[0]))
            return subconfig[".".join(keys[1:])]
        # Set in current configuration
        else:
            return getattr(self, keys[0])

    def add_option(self, key, option_type, default=NOT_SET, env_var=None):
        """Add a configuration setting.

        Parameters
        ----------
        key : str
            The name of the configuration setting. This must be a valid
            Python attribute name i.e. alphanumeric with underscores.
        option_type : function
            A function such as ``float``, ``int`` or ``str`` which takes
            the configuration value and returns an object of the correct
            type.  Note that the values retrieved from environment
            variables are always strings, while those retrieved from the
            YAML file might already be parsed. Hence, the function provided
            here must accept both types of input.
        default : object, optional
            The default configuration to return if not set. By default none
            is set and an error is raised instead.
        env_var : str, optional
            The environment variable name that holds this configuration
            value. If not given, this configuration can only be set in the
            YAML configuration file.

        """
        self._config[key] = {'type': option_type}
        if env_var is not None:
            self._config[key]['env_var'] = env_var
        if default is not NOT_SET:
            self._config[key]['default'] = default
