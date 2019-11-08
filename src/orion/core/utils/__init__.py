# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils` -- Package-wide useful routines
=======================================================

.. module:: utils
   :platform: Unix
   :synopsis: Helper functions useful in possibly all :mod:`orion.core`'s modules.
"""

from abc import ABCMeta
from collections import defaultdict
from glob import glob
from importlib import import_module
import logging
import os

import pkg_resources


log = logging.getLogger(__name__)


# Define type of arbitrary nested defaultdicts
def nesteddict():
    """Extend defaultdict to arbitrary nested levels."""
    return defaultdict(nesteddict)


class SingletonAlreadyInstantiatedError(ValueError):
    """Exception to be raised when someone provides arguments to build
    an object from a already-instantiated `SingletonType` class.
    """

    def __init__(self, name):
        """Pass the same constant message to ValueError underneath."""
        super().__init__("A singleton instance of (type: {}) has already been instantiated."
                         .format(name))


class SingletonNotInstantiatedError(TypeError):
    """Exception to be raised when someone try to access an instance
    of a singleton that has not been instantiated yet
    """

    def __init__(self, name):
        """Pass the same constant message to TypeError underneath."""
        super().__init__('No singleton instance of (type: {}) was created'
                         .format(name))


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


class Factory(ABCMeta):
    """Instantiate appropriate wrapper for the infrastructure based on input
    argument, ``of_type``.

    Attributes
    ----------
    types : list of subclasses of ``cls.__base__``
       Updated to contain all possible implementations currently. Check out code.
    typenames : list of str
       Names of implemented wrapper classes, correspond to possible ``of_type``
       values.

    """

    def __init__(cls, names, bases, dictionary):
        """Search in directory for attribute names subclassing `bases[0]`"""
        super(Factory, cls).__init__(names, bases, dictionary)

        cls.modules = []
        base = import_module(cls.__base__.__module__)
        try:
            py_files = glob(os.path.abspath(os.path.join(base.__path__[0], '[A-Za-z]*.py')))
            py_mods = map(lambda x: '.' + os.path.split(os.path.splitext(x)[0])[1], py_files)
            for py_mod in py_mods:
                cls.modules.append(import_module(py_mod, package=cls.__base__.__module__))
        except AttributeError:
            # This means that base class and implementations reside in a module
            # itself and not a subpackage.
            pass

        # Get types advertised through entry points!
        for entry_point in pkg_resources.iter_entry_points(cls.__name__):
            entry_point.load()
            log.debug("Found a %s %s from distribution: %s=%s",
                      entry_point.name, cls.__name__,
                      entry_point.dist.project_name, entry_point.dist.version)

        # Get types visible from base module or package, but internal
        def get_all_subclasses(parent):
            """Get set of subclasses recursively"""
            subclasses = set()
            for subclass in parent.__subclasses__():
                subclasses.add(subclass)
                subclasses |= get_all_subclasses(subclass)

            return subclasses

        cls.types = list(get_all_subclasses(cls.__base__))
        cls.types = [class_ for class_ in cls.types if class_.__name__ != cls.__name__]
        cls.typenames = list(map(lambda x: x.__name__.lower(), cls.types))
        log.debug("Implementations found: %s", cls.typenames)

    def __call__(cls, of_type, *args, **kwargs):
        """Create an object, instance of ``cls.__base__``, on first call.

        :param of_type: Name of class, subclass of ``cls.__base__``, wrapper
           of a database framework that will be instantiated on the first call.
        :param args: positional arguments to initialize ``cls.__base__``'s instance (if any)
        :param kwargs: keyword arguments to initialize ``cls.__base__``'s instance (if any)

        .. seealso::
           `Factory.typenames` for values of argument `of_type`.

        .. seealso::
           Attributes of ``cls.__base__`` and ``cls.__base__.__init__`` for
           values of `args` and `kwargs`.

        .. note:: New object is saved as `Factory`'s internal state.

        :return: The object which was created on the first call.
        """
        for inherited_class in cls.types:
            if inherited_class.__name__.lower() == of_type.lower():
                return inherited_class.__call__(*args, **kwargs)

        error = "Could not find implementation of {0}, type = '{1}'".format(
            cls.__base__.__name__, of_type)
        error += "\nCurrently, there is an implementation for types:\n"
        error += str(cls.typenames)
        raise NotImplementedError(error)


class SingletonFactory(AbstractSingletonType, Factory):
    """Wrapping `Factory` with `SingletonType`. Keep compatibility with `AbstractSingletonType`."""

    pass
