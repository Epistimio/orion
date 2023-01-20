""" Immutable dataclass containing the experiment configuration. """
from __future__ import annotations

import datetime  # noqa
import typing
from typing import Any

from typing_extensions import TypedDict

if typing.TYPE_CHECKING:
    from typing_extensions import Required

    from orion.core.evc.adapters import CompositeAdapter


class Refers(TypedDict):
    """Value of the 'refers' property of the Experiment."""

    parent_id: str | None
    root_id: int | str | None
    adapter: CompositeAdapter


# Note: total=false, because the attribute defaults to {} in Experiment __init__.


class RefersConfig(TypedDict, total=False):
    """Config for the 'refers' attribute of the Experiment."""

    parent_id: str | None
    root_id: int | str | None
    adapter: list[dict]


class MetaData(TypedDict, total=False):
    """Config for the 'metadata' property of the experiment."""

    # NOTE: Fields that are set in resolve_config when using the commandline interface:

    user: Required[str]
    """ System user currently owning this running process, the one who invoked **Oríon**. """

    orion_version: Required[str]
    """ Version of **Oríon** which suggested this experiment. `user`'s current **Oríon** version.
    """

    datetime: Required[datetime.datetime]
    """ When was this particular configuration submitted to the database. """

    user_script: str
    """ Full absolute path to `user`'s executable. """

    user_args: list[str]
    """ Contains separate arguments to be passed when invoking `user_script`, possibly templated
    for **Oríon**.
    """

    user_vcs: str | None
    """ User's version control system for this executable's code repository. """

    user_version: str | None
    """ Current user's repository version. """

    user_commit_hash: str | None
    """ Current `Experiment`'s commit hash for **Oríon**'s invocation. """


class ExperimentConfig(TypedDict):
    """TypedDict for the configuration of an `Experiment`.

    NOTE: The `Unpack` annotation can also be used to annotate the **kwargs of functions that
    expect to receive items of the experiment config as keyword arguments.
    For instance: `**exp_config: Unpack[ExperimentConfig]`.
    """

    name: str
    """ Unique identifier for this experiment per ``user``. """

    _id: int | str | None  # pylint: disable=invalid-name
    """ id of the experiment in the database if experiment is configured, or ``None`` if the
    experiment is not configured.
    """

    refers: RefersConfig  # todo: outdated docstring?
    """ A dictionary pointing to a past `Experiment` id, ``refers[parent_id]``, whose
    trials we want to add in the history of completed trials we want to re-use.
    For convenience and database efficiency purpose, all experiments of a common tree shares
    ``refers[root_id]``, with the root experiment referring to itself.
    """

    version: int
    """ Current version of this experiment. """

    metadata: MetaData
    """ Contains managerial information about this `Experiment`. """

    max_trials: int | None
    """ How many trials must be evaluated, before considering this `Experiment` done.
    This attribute can be updated if the rest of the experiment configuration
    is the same. In that case, if trying to set to an already set experiment,
    it will overwrite the previous one.
    """

    max_broken: int | None
    """How many trials must be broken, before considering this `Experiment` broken.
    This attribute can be updated if the rest of the experiment configuration
    is the same. In that case, if trying to set to an already set experiment,
    it will overwrite the previous one.
    """

    space: dict[str, Any]
    """ Object representing the optimization space. """

    algorithm: dict[str, Any] | None
    """ Complete specification of the optimization and dynamical procedures taking place in the
    Experiment.
    """

    working_dir: str | None
    """ Working directory. """

    knowledge_base: dict[str, Any] | None
    """ Configuration of the `KnowledgeBase`, if any. """


class PartialExperimentConfig(ExperimentConfig, total=False):
    """TypedDict for a partial configuration of an `Experiment`.

    NOTE: This can be used to annotate methods where only some of the keys of the Experiment's
    config are passed. For example, in `BaseStorageProtocol.update_experiment(**kwargs)`, the
    **kwargs can be annotated with Unpack[PartialExperimentConfig], so that we can typecheck the
    values that are passed as **kwargs to match the entries of the ExperimentConfig dict.
    Calls like update_experiment(foobar="bob") fail the type check.

    If the **kwargs were annotated with **Unpack[ExperimentConfig], then the type checker would
    expect every key of ExperimentConfig to be a required keyword argument of the function.

    See https://peps.python.org/pep-0692/ for more info on the Unpack annotation.
    """

    name: str
    _id: int | str | None  # pylint: disable=invalid-name
    refers: RefersConfig
    version: int
    metadata: MetaData
    max_trials: int | None
    max_broken: int | None
    space: dict[str, Any]
    algorithm: dict[str, Any] | None
    working_dir: str | None
    knowledge_base: dict[str, Any] | None
