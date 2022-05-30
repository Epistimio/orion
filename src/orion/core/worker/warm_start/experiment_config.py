from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)
from datetime import datetime
from logging import getLogger as get_logger
from .serializable import SerializableMixin

logger = get_logger(__name__)


@dataclass(frozen=True, unsafe_hash=True)
class Config(SerializableMixin):
    """Base class for the various Config objects: StorageConfig, ExperimentConfig, etc."""


@dataclass(frozen=True)
class DatabaseConfig(Config):
    type: str


@dataclass(frozen=True)
class StorageConfig(Config):
    type: str
    database: DatabaseConfig


@dataclass(frozen=True)
class Refers(Config):
    parent_id: Optional[str]
    root_id: int
    adapter: List[Any]


@dataclass(frozen=True)
class MetaData(Config):
    user: str
    """ System user currently owning this running process, the one who invoked **Oríon**. """

    orion_version: str
    """ Version of **Oríon** which suggested this experiment. `user`'s current **Oríon** version.
    """

    datetime: datetime
    """ When was this particular configuration submitted to the database. """

    # NOTE: Fields that don't appear to be set in the metadata, but that are mentioned in the
    # docstring:

    # user_script : str
    #    Full absolute path to `user`'s executable.
    # user_args : list of str
    #    Contains separate arguments to be passed when invoking `user_script`,
    #    possibly templated for **Oríon**.
    # user_vcs : str, optional
    #    User's version control system for this executable's code repository.
    # user_version : str, optional
    #    Current user's repository version.
    # user_commit_hash : str, optional
    #    Current `Experiment`'s commit hash for **Oríon**'s invocation.


@dataclass(frozen=True)
class ExperimentInfo(Config):
    """Immutable Dataclass that holds the information about an experiment."""

    name: str
    """ Unique identifier for this experiment per ``user``. """

    id: Optional[int]
    """ id of the experiment in the database if experiment is configured. Value is ``None`` if the
    experiment is not configured.
    """

    refers: Union[dict, List["ExperimentInfo"]]
    #    A dictionary pointing to a past `Experiment` id, ``refers[parent_id]``, whose
    #    trials we want to add in the history of completed trials we want to re-use.
    #    For convenience and database effiency purpose, all experiments of a common tree shares
    #    ``refers[root_id]``, with the root experiment refering to itself.

    version: int
    """ Current version of this experiment. """

    metadata: MetaData
    """ Contains managerial information about this `Experiment`. """

    max_trials: int
    """ How many trials must be evaluated, before considering this `Experiment` done.
    This attribute can be updated if the rest of the experiment configuration
    is the same. In that case, if trying to set to an already set experiment,
    it will overwrite the previous one.
    """

    max_broken: int
    """How many trials must be broken, before considering this `Experiment` broken.
    This attribute can be updated if the rest of the experiment configuration
    is the same. In that case, if trying to set to an already set experiment,
    it will overwrite the previous one.
    """

    space: Dict[str, Any]
    """ Object representing the optimization space. """

    algorithms: Dict[str, Any]
    """ Complete specification of the optimization and dynamical procedures taking place in the
    Experiment.
    """

    working_dir: str
    """ Working directory. """

    _id: int
    """ ID of the experiment. """

    # IDEA: Store a reference to the Storage object associated with the experiment in
    # case we want to support multi-storage in the future.
    _storage: Optional[StorageConfig] = field(default=None, hash=False, repr=False)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
