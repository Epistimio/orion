""" Immutable dataclass containing the experiment configuration. """
from __future__ import annotations

import typing
from typing import Any, TypedDict

if typing.TYPE_CHECKING:
    from orion.core.evc.adapters import CompositeAdapter


class EmptyDict(TypedDict):
    """Empty dictionary."""


class DatabaseConfig(TypedDict):
    """Database configuration."""

    type: str


class StorageConfig(TypedDict):
    """Storage configuration."""

    type: str
    database: DatabaseConfig


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


class BasicMetaData(TypedDict):
    """Config for the 'metadata' property of the experiment."""

    user: str
    """ System user currently owning this running process, the one who invoked **Oríon**. """

    orion_version: str
    """ Version of **Oríon** which suggested this experiment. `user`'s current **Oríon** version.
    """

    datetime: datetime
    """ When was this particular configuration submitted to the database. """


class MetaData(BasicMetaData, total=False):
    """Metadata of the experiment. Also contains some optional keys."""

    # NOTE: Fields that don't appear to be set in the metadata, but that are mentioned in the
    # docstring:

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
    """Immutable Dataclass that holds the information about an experiment."""

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

    algorithms: dict[str, Any] | None
    """ Complete specification of the optimization and dynamical procedures taking place in the
    Experiment.
    """

    working_dir: str | None
    """ Working directory. """

    knowledge_base: dict[str, Any] | None
    """ Configuration of the `KnowledgeBase`, if any. """
