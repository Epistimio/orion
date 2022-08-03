"""
Parse command line arguments for Orion
======================================

Parsing and building of the command line input for using script.

Simplify the parsing of a command line by storing every values inside a `dict`
mapping the name of the argument to its value as a key-pair relation. Positional arguments
are stored in the format `_X` where `X` represent the index of that argument inside
the command line.

CmdlineParser provides an interface to parse command line arguments from input but also
templates to build it again as a list or an already formatted string.
"""
import os
from collections import OrderedDict


class CmdlineParser:
    """Simple class for commandline arguments parsing.

    `CmdlineParser` provides a simple interface to interpret commandline arguments
    and create new commandlines. This class exposes an interface to transform a list of strings
    representing the commandline in a dictionary of key-value conserving the order of
    the arguments, the value of named arguments, be it boolean, single-valued or multiple ones,
    as well as handling the definition of argument through the `=` sign.

    Attributes
    ----------
    arguments : dict
        Commandline arguments' name and value(s).
    template : list
        List of template-ready strings for formatting.

    See Also
    --------
    parse : Parse the list of string and defines the form of the template.

    Notes
    -----
    Subcommands are not supported and will be likely lead to bad interpretation of the commandline.
    Aggregation of single characters arguments is not supported yet. Ex: `-xzvf`

    """

    def __init__(self):
        """See `CmdlineParser` description"""
        # TODO Handle parsing twice.
        self.keys = {}
        self.arguments = OrderedDict()
        self._already_parsed = False
        self.template = []

    def get_state_dict(self):
        """Give state dict that can be used to reconstruct the parser"""
        return dict(
            arguments=list(map(list, self.arguments.items())),
            keys=list(map(list, self.keys.items())),
            template=self.template,
        )

    def set_state_dict(self, state):
        """Reset the parser based on previous state"""
        if state.get("keys") is None:
            # NOTE: To support experiments prior to 0.1.9
            state["keys"] = [(key, self._key_to_arg(key)) for key in state["arguments"]]
        self.keys = OrderedDict(state["keys"])
        self.arguments = OrderedDict(state["arguments"])
        self.template = state["template"]
        self._already_parsed = bool(self.template)

    def format(self, configuration):
        """Format the current template.

        Return a list of string where the arguments have been given the values inside
        the `configuration` argument.

        Parameters
        ----------
        configuration : dict
            Dictionary containing the keys and values for each argument of the commandline.

        Returns
        -------
        list
            A list resembling the one given to the `parse` method but with values of
            `configuration`.

        """
        # Arguments with spaces are broken if we use ' '.join(template).format().split(' ').
        # Hence we iterate over the list as-is and format on each item separately.
        formatted = []
        for item in self.template:
            if item.startswith("-"):
                formatted.append(item)
            elif (
                item.startswith("{")
                and item.endswith("}")
                and any(item == f"{{{key}}}" for key in configuration)
            ):
                # The argument has an entry with exactly matching name in the configuration.
                # Extract it from the configuration, rather than try to use `str.format`.
                # This solves bugs that arise from using strings that are invalid python expressions
                # (e.g. names with ".", "/", or ":").
                key = [key for key in configuration if item == f"{{{key}}}"][0]
                value = configuration[key]
                formatted.append(str(value))
            else:
                formatted.append(item.format(**configuration))

        return formatted

    def parse(self, commandline):
        """Parse the `commandline` argument.

        This function will go through the list of strings and create a dictionary
        mapping every argument to its value(s). It will also form a template to recreate such a
        list. If `commandline` is empty or the `parse` function has already been called, it
        will return the dictionary of the already-parsed arguments.

        Parameters
        ----------
        commandline : list
            List of string representing the commandline arguments.

        Returns
        -------
        dict
            Dictionary holding the values of every argument. The keys are the name of the arguments.

        Raises
        ------
        ValueError
            This exception is raised then the parser detects a duplicate argument.
        RuntimeError
            This exception is raised if the parser already parsed a commandline and contains a
            template.

        Notes
        -----
        Each argument is stored inside a `dict`.
        Inside that dictionary, the keys are created following these rules:

        -If the argument is a positional one, its key will be `'_pos_x'` where `x` is its index
        inside the list.
        -Otherwise, the key will be equal to the name of the argument minus the prefixed dashes.

        The values are stored in the following ways:

        -If the argument is a positional one, its value is simply its value inside the commandline.
        -If the argument is a named, boolean argument, its value is simply `True`.
        -If the argument is a named, single-valued argument, its value is stored as-is.
        -If the argument is a named, multi-valued argument, its value is a list containing each
        value following it until the next named argument.

        Positional arguments following a named argument are not currently supported.

        Examples
        --------
        Positional and named arguments:

        >>> parser = CmdlineParser()
        >>> parser.parse('python 1 --arg value'.split(' '))
        {'_pos_0': 'python', '_pos_1': '1', 'arg': 'value'}

        Named boolean argument:

        >>> parser.parse('python --boolean'.split(' '))
        {'_pos_0': 'python', 'boolean': True}

        Named multi-valued argument:

        >>> parser.parse('python --args value1 value2'.split(' '))
        {'_pos_0': 'python', 'args': ['value1', 'value2']}

        Named argument defined with `=`:

        >>> parser.parse('python --arg=value'.split(' '))
        {'_pos_0': 'python', 'arg': 'value'}

        """
        if self._already_parsed:
            raise RuntimeError("The commandline has already been parsed.")

        keys, arguments = self._parse_arguments(commandline)
        self.arguments = arguments
        self.keys = keys

        for key, value in arguments.items():
            # Handle positional arguments
            if key.startswith("_"):
                self.template.append("{" + key + "}")

            # Handle optional ones
            else:
                arg = self.keys[key]
                if arg in self.template:
                    continue

                self.template.append(arg)

                # Ignore value as key is a boolean argument
                if isinstance(value, bool):
                    continue

                if not isinstance(value, list):
                    template = "{" + key + "}"
                    self.template.append(template)
                    continue

                for pos in range(len(value)):
                    template = "{" + key + "[" + str(pos) + "]}"
                    self.template.append(template)

        self._already_parsed = True

        return self.arguments

    @staticmethod
    def _key_to_arg(arg):
        if len(arg) > 1:
            return "--" + arg

        return "-" + arg

    def _parse_arguments(self, commandline):

        arguments = OrderedDict()
        keys = OrderedDict()
        key = None

        for item in commandline:
            # Handle keyworded arguments
            if item.startswith("-"):
                # Make sure we're not defining the same argument twice
                # If the argument is in the form of `--name=value`
                argument_parts = item.split("=")
                argument_name = argument_parts[0]
                key = argument_name.lstrip("-")

                if key in keys:
                    raise ValueError(
                        f"Conflict: two arguments have the same name: {key}"
                    )

                arguments[key] = []
                keys[key] = argument_name

                if len(argument_parts) > 1:
                    arguments[key].append(argument_parts[-1])

            # If the argument did not start with `-` but we have an argument name
            # That means that this value belongs to that argument name list
            elif key is not None and item.strip(" "):
                arguments[key].append(item)

            # No argument name means we have not reached them yet, so we're still in the
            # Positional arguments part
            elif key is None:
                _pos_key = f"_pos_{len(arguments)}"
                keys[_pos_key] = _pos_key
                arguments[_pos_key] = item

        for key, value in arguments.items():
            # Loop through the items and check if their value is a list
            # If it is, and the length is 0, that means it is a boolean args.
            # If its value is 1, it only has a single element and we unpack it.
            if isinstance(value, list):
                if not value:
                    value = True
                elif len(value) == 1:
                    value = value[0]

            value = self._parse_paths(value)
            arguments[key] = value

        return keys, arguments

    def _parse_paths(self, value):
        if isinstance(value, list):
            return [self._parse_paths(item) for item in value]

        if isinstance(value, str) and os.path.exists(value):
            return os.path.abspath(value)

        return value
