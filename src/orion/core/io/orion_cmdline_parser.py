# -*- coding: utf-8 -*-
"""
Parse command line arguments for Orion
======================================

Parsing and building of the command line input for using script.

Simplify the parsing of a command line by storing every values inside an `OrderedDict`
mapping the name of the argument to its value as a key-pair relation. Positional arguments
are stored in the format `_X` where `X` represent the index of that argument inside
the command line.

CmdlineParser provides an interface to parse command line arguments from input but also
utility functions to build it again as a list or an already formatted string.
"""

import copy
import errno
import logging
import os
import re
import shutil
from collections import OrderedDict, defaultdict

from orion.core.io.cmdline_parser import CmdlineParser
from orion.core.io.convert import infer_converter_from_file_type

log = logging.getLogger(__name__)


def _is_nonprior_wave(arg):
    return arg.startswith("/") or arg == ""


class OrionCmdlineParser:
    """Python interface commandline parser

    The `OrionCmdlineParser` is used as a way to obtain the parsing process of Orion
    through a Python interface. It provides different dictionaries containing information
    for different parts of the parsing process. It also exposes methods to retrieve the
    command line with formatted values for parameters.

    Parameters
    ----------
    config_prefix : str, optional
        Prefix for the configuration file used by the parser to identify it.
    allow_non_existing_files : bool, optional
        If True, will parse all commandline but ignore non existing user script or configuration
        files if non existant. Default is False

    Attributes
    ----------
    parser : CmdlineParser
        Parser that will be used to parse the commandline.
    cmd_priors : OrderedDict
        An OrderedDict containing only the priors inside the commandline.
    file_priors : OrderedDict
        An OrderedDict obtained by parsing the config file, if one was found.
    priors : OrderedDict
        An OrderedDict obtained from merging `cmd_priors` and `file_priors`.
    user_script : str
        File path of the script executed (inferred from parsed commandline)
    config_prefix : str
        Prefix for the configuration file used by the parser to identify it.
    file_config_path : str
        If a config file was found, this file contain the path. `None` otherwise.
    converter : BaseConverter
        A BaseConverter object to parse the config file.

    Methods
    -------
    parse(commandline)
        Parses the commandline and populate the OrderedDict
    format(trial, experiment) : str
        Return the commandline with replaced values for priors and attributes

    """

    def __init__(self, config_prefix="config", allow_non_existing_files=False):
        """Create an `OrionCmdlineParser`."""
        self.parser = CmdlineParser()
        self.cmd_priors = OrderedDict()
        self.file_priors = OrderedDict()
        self.config_file_data = {}

        self.config_prefix = config_prefix
        self.file_config_path = None
        self.converter = None

        self.allow_non_existing_files = allow_non_existing_files
        self.user_script = None

        # Extraction methods for the file parsing part.
        self._extraction_method = {
            dict: self._extract_dict,
            defaultdict: self._extract_defaultdict,
            list: self._extract_list,
            str: self._extract_file_string,
        }

        # Look for anything followed by a tilt and possible branching attributes + prior
        self.prior_regex = re.compile(r"(.+)~([\+\-\>]?.+)")

    def get_state_dict(self):
        """Give state dict that can be used to reconstruct the parser"""
        return dict(
            parser=self.parser.get_state_dict(),
            cmd_priors=list(map(list, self.cmd_priors.items())),
            file_priors=list(map(list, self.file_priors.items())),
            config_file_data=self.config_file_data,
            config_prefix=self.config_prefix,
            file_config_path=self.file_config_path,
            converter=self.converter.get_state_dict() if self.converter else None,
        )

    def set_state_dict(self, state):
        """Reset the parser based on previous state"""
        self.parser.set_state_dict(state["parser"])

        self.cmd_priors = OrderedDict(state["cmd_priors"])
        self.file_priors = OrderedDict(state["file_priors"])

        self.config_file_data = state["config_file_data"]
        self.config_prefix = state["config_prefix"]
        self.file_config_path = state["file_config_path"]

        if self.file_config_path:
            self.converter = infer_converter_from_file_type(self.file_config_path)
            self.converter.set_state_dict(state["converter"])

        if "user_script" in state:
            self.user_script = state["user_script"]
        else:
            self.infer_user_script(self.parser.format(self.parser.arguments))

    def parse(self, commandline):
        """Parse the commandline given for the definition of priors.

        Parse the commandline for priors and check if a specific key is found to parse
        an additional configuration file. Then the definition of the priors are stored
        inside the `priors` attribute.

        Raises
        ------
        ValueError
            If a prior inside the commandline and the config file have the same name.

        """
        self.infer_user_script(commandline)
        replaced = self._replace_priors(commandline)
        configuration = self.parser.parse(replaced)
        self._build_priors(configuration)

        duplicated_priors = set(self.cmd_priors.keys()) & set(self.file_priors.keys())
        if duplicated_priors:
            raise ValueError(
                "Conflict: definition of same prior in commandline and config: "
                "{}".format(duplicated_priors)
            )

    def infer_user_script(self, user_args):
        """Infer the script name and perform some checks"""
        if not user_args:
            return

        # TODO: Parse commandline for any options to python and pick the script filepath properly
        if user_args[0] == "python":
            user_script = user_args[1]
        else:
            user_script = user_args[0]

        if (
            not os.path.exists(user_script)
            and not shutil.which(user_script)
            and not self.allow_non_existing_files
        ):
            raise OSError(
                errno.ENOENT,
                "The path specified for the script does not exist",
                user_script,
            )

        self.user_script = user_script

    @property
    def priors(self):
        """Return an OrderedDict obtained from merging `cmd_priors` and `file_priors`."""
        priors = copy.deepcopy(self.file_priors)
        priors.update(self.cmd_priors)
        return priors

    @staticmethod
    def _replace_priors(args):
        """Change directly name priors to more general form.

        Pass through the current commandline arguments and replace priors of the form
        `--<name>~<prior>` to the more general one `--<name> orion~<prior>` to stay consistent
        with parameters passed as list.

        Parameters
        ----------
        args: list
            A list of the current commandline arguments.

        Returns
        -------
        list
            A list composed of the same elements as `args` augmented with the new form
            of the priors.

        """
        replaced = []
        for item in args:
            if item.startswith("-"):
                # Get the prior part after the `~`
                parts = item.split("~")

                if len(parts) > 1 and _is_nonprior_wave(parts[1]):
                    replaced.append(item)
                    continue

                # If the argument was defined has a long one but only has a single letter
                # then it needs to be shortened.
                if parts[0].startswith("--") and len(parts[0]) == 3:
                    parts[0] = parts[0][1:]

                replaced.append(parts[0])

                if len(parts) > 1:
                    replaced.append("orion~" + parts[1])
            else:
                replaced.append(item)

        return replaced

    def _build_priors(self, configuration):
        """Create OrderedDict from priors only.

        Loop through every commandline arguments and check if it might correspond to a prior or a
        configuration file. Configuration file is parsed with `_load_config` while cmdline priors
        are extracted with `_extract_prior`.

        Parameters
        ----------
        configuration: OrderedDict
            The original configuration from which to extract OrderedDict.

        """
        for key, value in configuration.items():
            if key == self.config_prefix:
                self.file_config_path = value
                self._load_config(value)
            else:
                self._extract_prior(key, value, self.cmd_priors)

    def _load_config(self, path):
        """Load configuration file.

        Load the configuration file associated with the `config` key. Will try to resolve a
        valid converter for the file extension (yaml, json or other). Content will be put
        inside the `self.file_priors` attribute. Once the data has been parsed from the file,
        the corresponding configuration will be created.

        Parameters
        ----------
        path: string
            Path to the configuration file.

        """
        if not os.path.exists(path):
            if self.allow_non_existing_files:
                log.info(
                    "The path specified for the script config does not exist: %s", path
                )
                return
            else:
                raise OSError(
                    errno.ENOENT,
                    "The path specified for the script config does not exist",
                    path,
                )

        self.converter = infer_converter_from_file_type(path)
        self.config_file_data = self.converter.parse(path)
        self._extraction_method[type(self.config_file_data)]("", self.config_file_data)

    def _extract_defaultdict(self, current_depth, ex_dict):
        for key, value in ex_dict.items():
            sub_depth = current_depth + "/" + str(key)

            try:
                if isinstance(value, str):
                    value = "orion~" + value
                self._extraction_method[type(value)](sub_depth, value)
            except KeyError:
                pass

    def _extract_dict(self, current_depth, ex_dict):
        """Recursively extract data from dictionary.

        Loop through the pairs of key/value inside `ex_dict` to find information regarding
        priors defined inside the config file.

        Parameters
        ----------
        current_depth: string
            String corresponding to the namespace at the current depth.
        ex_dict: dict
            Dictionary to loop through.

        """
        for key, value in ex_dict.items():
            sub_depth = current_depth + "/" + str(key)

            try:
                self._extraction_method[type(value)](sub_depth, value)
            except KeyError:
                pass

    def _extract_list(self, current_depth, ex_list):
        """Recursively extract data from list.

        Loop through the values inside `ex_list` to find information regarding priors defined
        inside the config file.

        Parameters
        ----------
        current_depth: string
            String corresponding to the namespace at the current depth.
        ex_list: list
            List to loop through.

        """
        for i, value in enumerate(ex_list):
            sub_depth = current_depth + "/" + str(i)

            try:
                self._extraction_method[type(value)](sub_depth, value)
            except KeyError:
                pass

    def _extract_file_string(self, current_depth, value):
        """Extract the prior from a string

        This is used alongside `_extract_list` and `_extract_dict` to iterate through
        a config file and extract the prior corresponding to a string.

        Parameters
        ----------
        current_depth: string
            String corresponding to the namespace at the current depth.
        value: string
            Value from which to extract a prior.

        """
        substrings = value.split("~")

        if len(substrings) == 1:
            return

        self._extract_prior(current_depth, value, self.file_priors)

    def _extract_prior(self, key, value, insert_into):
        """Insert parameters if it has a prior.

        Match the regex for priors with the `value` argument to extract the information
        regarding the prior. If it posseses such information, insert the parameters inside
        the `cmd_priors` attribute.

        Parameters
        ----------
        key: str
            Current key for the element inside the `OrderedDict` of commandline arguments.
        Will correspond to `orion` if it is related to priors.

        value:
            Possible parameter to parse through the regex.

        insert_into: OrderedDict
            Collections into which to insert the current prior.

        """
        if not isinstance(value, str):
            return

        prior = self.prior_regex.match(value)
        if prior is None:
            return

        # Skip first group because it will always correspond to `orion`.
        _, expression = prior.groups(2)

        name = key
        if not name.startswith("/"):
            name = "/" + name

        insert_into[name] = expression

    def _should_not_be_built(self, expression):
        """Check if an expression should be built or not.

        When parsing priors, we might encounter ones that use the conflicts solving notation.
        Some of these tokens need to be removed (like the `+` sign) so that the prior can be built
        whilst other must not be built because they do not add another dimension to the Space.
        Such tokens are: `-` and `>`.

        Parameters
        ----------
        expression: str
            The expression to be evaluated.

        """
        for token in self._invalid_priors_tokens:
            if expression.startswith(token):
                return True

        return False

    def format(self, config_path=None, trial=None, experiment=None):
        """Create the commandline for the user's script.

        Recreate the commandline passed to Orion for the user's script by replacing the instances
        of priors' expression by their actual value inside the `trial`. If a `config_path` is given,
        use the config file template to generate a new temporary one. Any templated argument of the
        form `{trial.xxx}` or `{exp.xxx}` will be replaced by their corresponding value.

        Parameters
        ----------
        config_path: str
            Path to the temporary config file. Must be given if the parser has a `file_config_path`.
        trial: `orion.core.worker.trial.Trial`
            A `Trial` object containing the values for the priors.
        experiment: `orion.core.worker.experiment.Experiment`
            An `Experiment` object containing information relative to the `trial`'s experiment.

        Returns
        -------
        list
            The commandline arguments.

        Raises
        ------
        ValueError
            If the configuration contains a config file but `format()` is called without the
            argument `config_path`.

        """
        if self.file_config_path and config_path is None:
            raise ValueError(
                "The configuration contains a config file. "
                "Cannot format without a `config_path` argument."
            )
        elif self.file_config_path:
            self._create_config_file(config_path, trial)
        configuration = self._build_configuration(trial)

        if config_path is not None:
            configuration[self.config_prefix] = config_path

        templated = self.parser.format(configuration)

        trial_and_exp = dict(trial=trial, exp=experiment)

        for idx, item in enumerate(templated):
            templated[idx] = item.format(**trial_and_exp)

        return templated

    def _create_config_file(self, config_path, trial):
        # Create a copy of the template
        instance = copy.deepcopy(self.config_file_data)

        for name, value in trial.params.items():
            # The param will only correspond to config keyd
            # that require a prior, so we make sure to skip
            # the ones that do not.
            if name not in self.file_priors.keys():
                continue

            # Since namespace start with '/', we must skip
            # the first element of the list.
            path = name.split("/")[1:]
            current_depth = instance

            for key in path:
                # If we meet a list, the key might correspond
                # to the index of a dictionary in that list
                if isinstance(current_depth, list):
                    if not key.isdigit():
                        continue

                    key = int(key)

                    # Make sure the key is not out of bound
                    try:
                        current_depth[key]
                    except IndexError:
                        break

                if isinstance(current_depth[key], str):
                    current_depth[key] = value
                else:
                    current_depth = current_depth[key]

        self.converter.generate(config_path, instance)

    def _build_configuration(self, trial):
        configuration = copy.deepcopy(self.parser.arguments)

        for name, value in trial.params.items():
            name = name.lstrip("/")
            configuration[name] = value

        return configuration

    def priors_to_normal(self):
        """Remove the namespace `/` prefix from priors."""
        return {key.lstrip("/"): arg for key, arg in self.cmd_priors.items()}
