# -*- coding: utf-8 -*-
"""
Singleton helpers and boilerplate
=================================

"""
from abc import ABCMeta

from orion.core.utils import Factory


class SingletonAlreadyInstantiatedError(ValueError):
    """Exception to be raised when someone provides arguments to build
    an object from a already-instantiated `SingletonType` class.
    """

    def __init__(self, name):
        """Pass the same constant message to ValueError underneath."""
        super().__init__(
            "A singleton instance of (type: {}) has already been instantiated.".format(
                name
            )
        )


class SingletonNotInstantiatedError(TypeError):
    """Exception to be raised when someone try to access an instance
    of a singleton that has not been instantiated yet
    """

    def __init__(self, name):
        """Pass the same constant message to TypeError underneath."""
        super().__init__("No singleton instance of (type: {}) was created".format(name))


class SingletonType(type):
    """Metaclass that implements the singleton pattern for a Python class."""

    def __init__(cls, name, bases, dictionary):
        """Create a class instance variable and initiate it to None object."""
        super(SingletonType, cls).__init__(name, bases, dictionary)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        """Create an object if does not already exist, otherwise return what there is."""
        if cls.instance is None:
            try:
                cls.instance = super(SingletonType, cls).__call__(*args, **kwargs)
            except TypeError as exception:
                raise SingletonNotInstantiatedError(cls.__name__) from exception

        elif args or kwargs:
            raise SingletonAlreadyInstantiatedError(cls.__name__)

        return cls.instance


class AbstractSingletonType(SingletonType, ABCMeta):
    """This will create singleton base classes, that need to be subclassed and implemented."""

    pass


class SingletonFactory(AbstractSingletonType, Factory):
    """Wrapping `orion.core.utils.Factory` with `SingletonType`. Keep compatibility with
    `AbstractSingletonType`."""

    pass


def update_singletons(values=None):
    """Replace singletons by given values and return previous singleton objects"""
    if values is None:
        values = {}

    # Avoiding circular import problems when importing this module.
    from orion.core.io.database import Database
    from orion.core.io.database.ephemeraldb import EphemeralDB
    from orion.core.io.database.mongodb import MongoDB
    from orion.core.io.database.pickleddb import PickledDB
    from orion.storage.base import Storage
    from orion.storage.legacy import Legacy
    from orion.storage.track import Track

    singletons = (Storage, Legacy, Database, MongoDB, PickledDB, EphemeralDB, Track)

    updated_singletons = {}
    for singleton in singletons:
        updated_singletons[singleton] = singleton.instance
        singleton.instance = values.get(singleton, None)

    return updated_singletons
