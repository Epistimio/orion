# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.orion_cmdline_parser` -- Parse command line arguments for Orion
===================================================================================

.. module:: orion_cmdline_parser
   :platform: Unix
   :synopsis: Parsing and building of the command line input for using script

Simplify the parsing of a command line by storing every values inside an `OrderedDict`
mapping the name of the argument to its value as a key-pair relation. Positional arguments
are stored in the format `_X` where `X` represent the index of that argument inside
the command line.

CmdlineParser provides an interface to parse command line arguments from input but also
utility functions to build it again as a list or an already formatted string.
"""
import copy

from collections import OrderedDict

from orion.core.io.cmdline_parser import CmdlineParser
from orion.core.io.convert import infer_converter_from_file_type

import re


class OrionCmdlineParser():

    def __init__(self, config_prefix='--config', parsing_tokens=None):
        self.parser = CmdlineParser(parsing_tokens=parsing_tokens)
        self.original = OrderedDict()
        self.priors_only = OrderedDict()
        self.file_config = OrderedDict()
        self.augmented_config = OrderedDict()

        self.config_prefix = config_prefix
        self.file_config_path = None
        self.converter = None

        self._extraction_method = {dict: self._extract_dict,
                                   list: self._extract_list,
                                   str: self._extract_file_string}

        # Look for anything followed by a tilt and possible branching attributes + prior
        self.prior_regex = re.compile(r'(.+)~([\+\-\>]?.+)')

    def parse(self, commandline):
        self.original = self.parser.parse(commandline)

        replaced = self._replace_priors(commandline)
        configuration = self.parser.parse(replaced)
        self._build_priors_only(configuration)

        self.augmented_config = copy.deepcopy(self.file_config)
        self.augmented_config.update(self.priors_only)

    def _replace_priors(self, args):
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
            if item.startswith('-'):
                parts = item.split('~')
                replaced.append(parts[0])

                if len(parts) > 1:
                    replaced.append('orion~' + parts[1])
            else:
                replaced.append(item)

        return replaced

    def _build_priors_only(self, configuration):
        """Create OrderedDict from priors only.

        Loop through every commandline arguments and check if it might correspond the a prior.
        If it does, extract the name and expression from it and insert them into the corresponding
        OrderedDict.

        Parameters
        ----------
        configuration: OrderedDict
            The original configuration from which to extract OrderedDict.
        """
        for key, value in configuration.items():
            if isinstance(value, list):
                for item in value:
                    self._extract_prior(item, self.priors_only)
            else:
                if key == self.config_prefix:
                    self.file_config_path = value
                    self._load_config(value)
                else:
                    self._extract_prior(value, self.priors_only)

    def _load_config(self, path):
        """Load configuration file.

        Load the configuration file associated with the `config` key. Will try to resolve a
        valid converter for the file extension (yaml, json or other). Content will be put
        inside the `self.file_config` attribute. Once the data has been parsed from the file,
        the corresponding configuration will be created.

        Parameters
        ----------
        path: string
            Path to the configuration file.
        """
        self.converter = infer_converter_from_file_type(path)
        self.file_config = self.converter.parse(path)
        self._extraction_method[type(self.file_config)]("", self.file_config)

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
            sub_depth = current_depth + '/' + str(key)

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
            sub_depth = current_depth + '/' + str(i)

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
        substrings = value.split('~')

        if len(substrings) == 1:
            return

        righthand_side = '~'.join(substrings[1:])
        expression = current_depth + '~' + righthand_side
        self._extract_prior(expression, self.file_config)

    def _extract_prior(self, value, insert_into):
        """Insert parameters if it has a prior.

        Match the regex for priors with the `value` argument to extract the information
        regarding the prior. If it posseses such information, insert the parameters inside
        the `priors_only` attribute.

        Parameters
        ----------
        key: string
            Current key for the element inside the `OrderedDict` of commandline arguments.
        Will correspond to `orion` if it is related to priors.

        value: string
            Possible parameter to parse through the regex.

        insert_into: OrderedDict
            Collections into which to insert the current prior.
        """
        prior = self.prior_regex.match(value)
        if prior is None:
            return

        name, expression = prior.groups(2)

        if not name.startswith('/'):
            name = '/' + name

        insert_into[name] = expression

    def format(self, trial, configuration):
        pass
