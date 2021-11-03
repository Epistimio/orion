# -*- coding: utf-8 -*-
"""
Singleton helpers and boilerplate
=================================

"""
from abc import ABCMeta

from orion.core.utils import Factory, GenericFactory


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
    from orion.core.io.database import database_factory
    from orion.storage.base import storage_factory

    singletons = (storage_factory, database_factory)

    updated_singletons = {}
    for singleton in singletons:
        name = singleton.base.__name__.lower()
        updated_singletons[name] = singleton.instance
        singleton.instance = values.get(name, None)

    return updated_singletons


class GenericSingletonFactory(GenericFactory):
    """Factory to create singleton instances of classes inheriting a given ``base`` class.

    .. seealso::

        :py:class:`orion.core.utils.GenericFactory`

    """

    def __init__(self, base):
        super(GenericSingletonFactory, self).__init__(base=base)
        self.instance = None

    def create(self, of_type=None, *args, **kwargs):
        """Create an object, instance of ``self.base``

        If the instance is already created, ``self.create`` can only be called without arguments
        and will return the singleton.

        Cannot be called without arguments if the singleton was not already created.

        Parameters
        ----------
        of_type: str, optional
            Name of class, subclass of ``self.base``. Capitalization insensitive.

        args: *
            Positional arguments to construct the givin class.

        kwargs: **
            Keyword arguments to construct the givin class.

        Raises
        ------
        `SingletonNotInstantiatedError`
            - If ``self.create()`` was never called and is called without arguments for the first
              time.
            - If ``self.create()`` was never called and the current call raises an error.
        `SingletonAlreadyInstantiatedError`
            If ``self.create()`` was already called with arguments (the singleton exist) and
            is called again with arguments.

        """

        if self.instance is None and of_type is None:
            raise SingletonNotInstantiatedError(self.base.__name__)
        elif self.instance is None:
            try:
                self.instance = super(GenericSingletonFactory, self).create(
                    of_type, *args, **kwargs
                )
            except TypeError as exception:
                raise SingletonNotInstantiatedError(self.base.__name__) from exception

        elif of_type or args or kwargs:
            raise SingletonAlreadyInstantiatedError(self.base.__name__)

        return self.instance
