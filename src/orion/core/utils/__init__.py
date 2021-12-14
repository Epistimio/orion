# -*- coding: utf-8 -*-
"""
Package-wide useful routines
============================

"""

import logging
import os
import signal
from abc import ABCMeta
from collections import defaultdict
from contextlib import contextmanager
from glob import glob
from importlib import import_module

import pkg_resources

log = logging.getLogger(__name__)


def nesteddict():
    """
    Define type of arbitrary nested defaultdicts
    Extend defaultdict to arbitrary nested levels.
    """
    return defaultdict(nesteddict)


def float_to_digits_list(number):
    """Convert a float into a list of digits, without conserving exponant"""
    # Get rid of scientific-format exponant
    str_number = str(number)
    str_number = str_number.split("e")[0]

    res = [int(ele) for ele in str_number if ele.isdigit()]

    # Remove trailing 0s in front
    while len(res) > 1 and res[0] == 0:
        res.pop(0)

    # Remove training 0s at end
    while len(res) > 1 and res[-1] == 0:
        res.pop(-1)

    return res


def get_all_subclasses(parent):
    """Get set of subclasses recursively"""
    subclasses = set()
    for subclass in parent.__subclasses__():
        subclasses.add(subclass)
        subclasses |= get_all_subclasses(subclass)

    return subclasses


def get_all_types(parent_cls, cls_name):
    """Get all subclasses and lowercase subclass names"""
    types = list(get_all_subclasses(parent_cls))
    types = [class_ for class_ in types if class_.__name__ != cls_name]

    return {class_.__name__.lower(): class_ for class_ in types}


def _import_modules(cls):
    cls.modules = []
    # TODO: remove?
    # base = import_module(cls.__base__.__module__)

    # Get types advertised through entry points!
    for entry_point in pkg_resources.iter_entry_points(cls.__name__):
        entry_point.load()
        log.debug(
            "Found a %s %s from distribution: %s=%s",
            entry_point.name,
            cls.__name__,
            entry_point.dist.project_name,
            entry_point.dist.version,
        )


def _set_typenames(cls):
    # Get types visible from base module or package, but internal
    cls.types.update(get_all_types(cls.__base__, cls.__name__))

    log.debug("Implementations found: %s", sorted(cls.types.keys()))


class GenericFactory:
    """Factory to create instances of classes inheriting a given ``base`` class.

    The factory can instantiate children of the base class at any level of inheritance.
    The children class must have different names (capitalization insensitive). To instantiate
    objects with the factory, use ``factory.create('name_of_the_children_class')`` passing the name
    of the children class to instantiate.

    To support classes even when they are not imported, register them in the ``entry_points``
    of the package's ``setup.py``. The factory will import all registered classes in the
    entry_points before looking for available children to create new objects.

    Parameters
    ----------
    base: class
       Base class of all children that the factory can instantiate.

    """

    def __init__(self, base):
        self.base = base

    def create(self, of_type, *args, **kwargs):
        """Create an object, instance of ``self.base``

        Parameters
        ----------
        of_type: str
            Name of class, subclass of ``self.base``. Capitalization insensitive

        args: *
            Positional arguments to construct the givin class.

        kwargs: **
            Keyword arguments to construct the givin class.
        """

        constructor = self.get_class(of_type)
        return constructor(*args, **kwargs)

    def get_class(self, of_type):
        """Get the class object (not instantiated)

        Parameters
        ----------
        of_type: str
            Name of class, subclass of ``self.base``. Capitalization insensitive
        """
        of_type = of_type.lower()
        constructors = self.get_classes()

        if of_type not in constructors:
            error = "Could not find implementation of {0}, type = '{1}'".format(
                self.base.__name__, of_type
            )
            error += "\nCurrently, there is an implementation for types:\n"
            error += str(sorted(constructors.keys()))
            raise NotImplementedError(error)

        return constructors[of_type]

    def get_classes(self):
        """Get children classes of ``self.base``"""
        _import_modules(self.base)
        return get_all_types(self.base, self.base.__name__)


class Factory(ABCMeta):
    """Deprecated, will be removed in v0.3.0. See GenericFactory instead"""

    def __init__(cls, names, bases, dictionary):
        super(Factory, cls).__init__(names, bases, dictionary)
        cls.types = {}
        try:
            _import_modules(cls)
        except ImportError:
            pass
        _set_typenames(cls)

    def __call__(cls, of_type, *args, **kwargs):
        """Create an object, instance of ``cls.__base__``, on first call."""
        _import_modules(cls)
        _set_typenames(cls)

        for name, inherited_class in cls.types.items():
            if name == of_type.lower():
                return inherited_class(*args, **kwargs)

        error = "Could not find implementation of {0}, type = '{1}'".format(
            cls.__base__.__name__, of_type
        )
        error += "\nCurrently, there is an implementation for types:\n"
        error += str(sorted(cls.types.keys()))
        raise NotImplementedError(error)


# pylint: disable = unused-argument
def _handler(signum, frame):
    log.error("Oríon has been interrupted.")
    raise KeyboardInterrupt


@contextmanager
def sigterm_as_interrupt():
    """Intercept ``SIGTERM`` signals and raise ``KeyboardInterrupt`` instead"""
    ## Signal only works inside the main process
    previous = signal.signal(signal.SIGTERM, _handler)

    yield None

    signal.signal(signal.SIGTERM, previous)
