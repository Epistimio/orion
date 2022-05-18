"""
Function utilities for evc in cli
=================================

Orion helper functions to parse command-line options for branching resolution

"""
from orion.core.evc import adapters
from orion.core.evc.conflicts import Resolution


def _add_enable_argument(parser):
    parser.add_argument(
        "--enable-evc",
        action="store_true",
        default=None,
        help="Enable the Experiment Version Control.",
    )


def _add_auto_resolution_argument(parser):
    parser.add_argument(
        "--auto-resolution",
        action="store_true",
        default=None,
        help="Deprecated. Conflicts are now resolved automatically by default."
        "See --manual-resolution to avoid auto-resolution.",
    )


def _add_manual_resolution_argument(parser):
    parser.add_argument(
        "--manual-resolution",
        action="store_true",
        default=None,
        help="Manually resolve conflicts",
    )


def _add_non_monitored_arguments_argument(parser):
    parser.add_argument(
        "--non-monitored-arguments",
        type=str,
        help="Ignore these arguments when looking for differences",
    )


def _add_ignore_code_changes_argument(parser):
    parser.add_argument(
        "--ignore-code-changes",
        action="store_true",
        default=None,
        help="Ignore code changes when looking for differences",
    )


def _add_branch_from_argument(parser):
    parser.add_argument(
        "--branch-from",
        type=str,
        help="Create a new child based on experiment defined by `branch-from`",
    )


def _add_algorithm_argument(parser, resolution_class):
    parser.add_argument(
        resolution_class.ARGUMENT,
        action="store_true",
        default=None,
        help="Set algorithm change as resolved if a branching event occur",
    )


def _add_code_argument(parser, resolution_class):
    parser.add_argument(
        resolution_class.ARGUMENT,
        choices=adapters.CodeChange.types,
        help="Set code change type",
    )


def _add_cli_argument(parser, resolution_class):
    parser.add_argument(
        resolution_class.ARGUMENT,
        choices=adapters.CommandLineChange.types,
        help="Set command line change type",
    )


def _add_config_argument(parser, resolution_class):
    parser.add_argument(
        resolution_class.ARGUMENT,
        choices=adapters.ScriptConfigChange.types,
        help="Set configuration change type",
    )


def _add_orion_version_argument(parser, resolution_class):
    parser.add_argument(
        resolution_class.ARGUMENT,
        action="store_true",
        default=None,
        help="Set orion version change as resolved if branching event occurred",
    )


def _add_branch_to_argument(parser, resolution_class):
    parser.add_argument(
        "-b",
        resolution_class.ARGUMENT,
        metavar="stringID",
        help="Unique name for the new branching experiment",
    )


resolution_arguments = {
    "enable": _add_enable_argument,
    "auto_resolution": _add_auto_resolution_argument,
    "manual_resolution": _add_manual_resolution_argument,
    "non_monitored_arguments": _add_non_monitored_arguments_argument,
    "ignore_code_changes": _add_ignore_code_changes_argument,
    "algorithm_change": _add_algorithm_argument,
    "code_change_type": _add_code_argument,
    "cli_change_type": _add_cli_argument,
    "config_change_type": _add_config_argument,
    "orion_version_change": _add_orion_version_argument,
    "branch_from": _add_branch_from_argument,
    "branch_to": _add_branch_to_argument,
}


UNDEFINED_PARSER_ERROR = (
    "A resolution with metavar '{}' is defined but no corresponding parser is defined. "
    "Please raise an issue on github if you encounter this error message."
)


def get_branching_args_group(parser):
    """Return the arguments for automatic branching resolution."""
    branching_args_group = parser.add_argument_group(
        "Oríon branching arguments (optional)",
        description="Arguments to automatically resolved branching events.",
    )

    _add_enable_argument(branching_args_group)
    _add_manual_resolution_argument(branching_args_group)
    _add_non_monitored_arguments_argument(branching_args_group)
    _add_ignore_code_changes_argument(branching_args_group)
    _add_auto_resolution_argument(branching_args_group)
    _add_branch_from_argument(branching_args_group)

    for resolution_class in sorted(
        Resolution.__subclasses__(), key=lambda cls: cls.__name__
    ):
        if not resolution_class.ARGUMENT:
            continue

        metavar = resolution_class.namespace()

        # This should fail blatantly if we forget to create a parser for a new resolution which
        # needs argument in command-line for automatic resolution.
        assert metavar in resolution_arguments, UNDEFINED_PARSER_ERROR.format(metavar)

        resolution_arguments[metavar](branching_args_group, resolution_class)

    return branching_args_group


def fetch_branching_configuration(config):
    """Build a dictionary of arguments for branching"""
    return {key: config[key] for key in resolution_arguments if key in config}
