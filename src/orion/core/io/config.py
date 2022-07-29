# pylint: disable=redefined-builtin
"""
Configuration object
====================

Configuration object to define package configuration.

Highly inspired from https://github.com/mila-iqia/blocks/blob/master/blocks/config.py.

"""
import contextlib
import logging
import os
import pprint

import yaml

logger = logging.getLogger(__name__)


NOT_SET = object()


@contextlib.contextmanager
def _disable_logger(disable=True):
    if disable:
        logger.disabled = True
    yield

    if disable:
        logger.disabled = False


class ConfigurationError(Exception):
    """Error raised when a configuration value is requested but not set."""


def _curate(key):
    return key.replace("-", "_")


class Configuration:
    """Configuration object

    Provides default values configurable at different levels. The configuration object can have
    global default values, which may be overridden by user with yaml configuration files,
    environment variables or by setting directly the values in the configuration object. In order,
    direct definition overrides, environment variables, which overrides yaml configuration, which
    overrides default values in configuration object definition.

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

    SPECIAL_KEYS = [
        "_config",
        "_subconfigs",
        "_yaml",
        "_default",
        "_env_var",
        "_help",
        "_deprecated",
    ]

    def __init__(self):
        self._config = {}
        self._subconfigs = {}

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
        with open(path, encoding="utf8") as f:
            cfg = yaml.safe_load(f)
            if cfg is None:
                return
            self._load_yaml_dict(self, cfg)

    def _load_yaml_dict(self, root, config):
        for key in self._config:
            if key not in config:
                continue
            value = config.pop(key)
            default = self[key + "._default"]
            deprecated = self[key + "._deprecated"]
            logger.debug('Overwritting "%s" default %s with %s', key, default, value)
            self[key + "._yaml"] = value
            if deprecated and deprecated.get("alternative"):
                logger.debug(
                    'Overwritting "%s" default %s with %s', key, default, value
                )
                root[deprecated.get("alternative") + "._yaml"] = value

        for key, item in self._subconfigs.items():
            if key not in config:
                continue

            # pylint: disable=protected-access
            item._load_yaml_dict(root, config.pop(key))

        if config:
            # Make it fail
            self[next(iter(config.keys()))]

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
        if key == "config":
            raise AttributeError

        if key not in self._config and key not in self._subconfigs:
            raise ConfigurationError(
                f"Configuration does not have an attribute '{key}'."
            )
        if key in self._subconfigs:
            return self._subconfigs[key]

        config_setting = self._config[key]
        if "value" in config_setting:
            value = config_setting["value"]
        elif "env_var" in config_setting and config_setting["env_var"] in os.environ:
            value = os.environ[config_setting["env_var"]]
            if config_setting["type"] in (list, tuple):
                value = value.split(":")
        elif "yaml" in config_setting:
            value = config_setting["yaml"]
        elif "default" in config_setting:
            value = config_setting["default"]
        else:
            raise ConfigurationError(
                f"Configuration not set and no default provided: {key}."
            )

        if config_setting.get("deprecated"):
            self._deprecate(key)

        return config_setting["type"](value)

    def __setattr__(self, key, value):
        """Set option value or subconfiguration

        Parameters
        ----------
        key: str
            The key or namespace to set the value of the configuration.
            If the configuration has subconfiguration, the key may be
            hierarchical with each level separated by dots.
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
        key = _curate(key)
        if key not in self.SPECIAL_KEYS and key in self._config:
            self._validate(key, value)
            self._config[key]["value"] = value
            if self._config[key].get("deprecated"):
                self._deprecate(key, value)

        elif key in ["_config", "_subconfigs"]:
            super().__setattr__(key, value)

        elif key in self._subconfigs:
            raise ValueError(f"Configuration already contains subconfiguration {key}")

        elif isinstance(value, Configuration):
            self._subconfigs[key] = value

        else:
            raise TypeError(
                f"Can only set {key} as a Configuration, not {type(value)}. "
                "Use add_option to set a new option."
            )

    # pylint: disable=unused-argument
    def _deprecate(self, key, value=NOT_SET):
        deprecate = self._config[key]["deprecated"]
        message = "(DEPRECATED) Option `%s` will be removed in %s."
        args = [deprecate.get("name", key), deprecate["version"]]
        if "alternative" in deprecate:
            message += " Use `%s` instead."
            args.append(deprecate["alternative"])

        logger.warning(message, *args)

    def get(self, key, deprecated="warn"):
        """Access value

        Parameters
        ----------
        key: str
            Key to access in the configuration. Similar to config.key.
        deprecated: str, optional
            If 'warn', the access to deprecated options will log a deprecation warning.
            else if 'ignore', no warning will be logged for access to deprecated options.

        """
        with _disable_logger(disable=(deprecated == "ignore")):
            value = self[key]

        return value

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
            raise TypeError(f"Cannot overwrite option {key} with a configuration")

        try:
            self._config[key]["type"](value)
        except ValueError as e:
            message = (
                f"Option {key} of type {self._config[key]['type']} "
                f"cannot be set to {value} with type {type(value)}"
            )
            raise TypeError(message) from e

    def __setitem__(self, key, value):
        """Set option value using dict-like syntax

        Parameters
        ----------
        key: str
            The key or namespace to set the value of the configuration.
            If the configuration has subconfiguration, the key may be
            hierarchical with each levels separated by dots.
            Ex: 'first.second.third'
        value: object
            A general object to set an option.

        """
        keys = list(map(_curate, key.split(".")))

        # Set in current config for special keys
        if len(keys) == 2 and keys[-1] in self.SPECIAL_KEYS:
            key, field = keys
            self._validate(key, value)
            self._config[key][field.lstrip("_")] = value
            if self._config[key].get("deprecated"):
                self._deprecate(key)

        # Set in current configuration
        elif len(keys) == 1:
            setattr(self, keys[0], value)

        # Recursively in sub configurations
        else:
            subconfig = getattr(self, keys[0])
            if subconfig is None:
                raise KeyError(f"'{keys[0]}' is not defined in configuration.")
            subconfig[".".join(keys[1:])] = value

    def __getitem__(self, key):
        """Get option value using dict-like syntax

        Parameters
        ----------
        key: str
            The key or namespace to set the value of the configuration.
            If the configuration has subconfiguration, the key may be
            hierarchical with each levels separated by dots.
            Ex: 'first.second.third'

        """
        keys = list(map(_curate, key.split(".")))

        # Recursively in sub configurations
        if len(keys) == 2 and keys[1] in self.SPECIAL_KEYS:
            key_config = self._config.get(keys[0], None)
            if key_config is None:
                raise ConfigurationError(
                    f"Configuration does not have an attribute '{keys[0]}'."
                )
            return key_config.get(keys[1][1:], None)
        elif len(keys) > 1:
            subconfig = getattr(self, keys[0])
            if subconfig is None:
                raise ConfigurationError(
                    f"Configuration does not have an attribute '{key}'."
                )
            return subconfig[".".join(keys[1:])]
        # Set in current configuration
        else:
            return getattr(self, keys[0])

    def add_option(
        self, key, option_type, default=NOT_SET, env_var=None, deprecate=None, help=None
    ):
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
        deprecate: `dict`, optional
            Should define dict(version, alternative), version at which the deprecated option will be
            removed and alternative to use. A deprecation warning will be logged each time this
            option is set by user. The option `name` can be used in addition to `version` and
            `alternative` to provide a different name then the key. This is useful if the key
            is in a subconfiguration and we want the deprecation error message to include the full
            path. This will add (DEPRECATED) at the beginning of the help message.
        help : str, optional
            Documentation for the option. Can be reused to build documentation
            or to build parsers with help messages.
            Default help message is 'Undocumented'.

        """
        key = _curate(key)
        if key in self._config or key in self._subconfigs:
            raise ValueError(f"Configuration already contains {key}")
        self._config[key] = {"type": option_type}
        if env_var is not None:
            self._config[key]["env_var"] = env_var
        if default is not NOT_SET:
            self._config[key]["default"] = default
        if deprecate is not None:
            if "version" not in deprecate:
                raise ValueError(
                    f"`version` is missing in deprecate option: {deprecate}"
                )
            self._config[key]["deprecated"] = deprecate

        if help is None:
            help = "Undocumented"

        if default is not NOT_SET:
            help += f" (default: {default})"
        if deprecate is not None:
            help = "(DEPRECATED) " + help
        self._config[key]["help"] = help

    def help(self, key):
        """Return the help message for the given option."""
        return self[key + "._help"]

    def add_arguments(self, parser, rename=None):
        """Add arguments to an `argparse` parser based on configuration

        This does not support subconfigurations. They will be ignored.

        Parameters
        ----------
        parser: `argparse.ArgumentParser`
            Parser to which this function will add arguments
        rename: dict, optional
            Mappings to provide different commandline names. Ex `{key: --my-arg}`

        """
        if rename is None:
            rename = {}

        for key, item in self._config.items():
            # TODO: Try with list and nargs='*', but it may case issues with
            # nargs=argparse.REMAINDER.
            if item["type"] in (dict, list, tuple):
                continue

            # NOTE: Do not set default, if parser.parse_argv().options[key] is None, then code
            # should look to config[key].
            arg_name = rename.get(key, f"--{key.replace('_', '-')}")
            parser.add_argument(
                arg_name,
                type=item["type"],
                help=item.get("help"),
            )

    def __contains__(self, key):
        """Return True if the option is defined."""
        return key in self._config or key in self._subconfigs

    def to_dict(self):
        """Return a dictionary representation of the configuration"""
        config = {}

        with _disable_logger():
            for key in self._config:
                config[key] = self[key]

            for key in self._subconfigs:
                config[key] = self[key].to_dict()

        return config

    def from_dict(self, config):
        """Set the configuration from a dictionary"""

        logger.debug("Setting config to %s", config)
        logger.debug("Config was %s", repr(self))

        with _disable_logger():
            for key in self._config:  # pylint: disable=consider-using-dict-items
                value = config.get(key, NOT_SET)

                if value is not NOT_SET:
                    self[key] = value
                else:
                    self._config[key].pop("value", None)

            for key in self._subconfigs:
                value = config.get(key)
                self[key].from_dict(value)

        logger.debug("Config is %s", repr(self))

    def __repr__(self) -> str:
        return pprint.pformat(self.to_dict())
