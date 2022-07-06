"""
Package-wide useful routines
============================

"""
from __future__ import annotations

import hashlib
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
    """Convert a float into a list of digits, without conserving exponent"""
    # Get rid of scientific-format exponent
    str_number = str(number)
    str_number = str_number.split("e", maxsplit=1)[0]

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
        assert entry_point.dist is not None
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


from typing import Generic, TypeVar

T = TypeVar("T")  # pylint: disable=invalid-name


class GenericFactory(Generic[T]):
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

    def __init__(self, base: type[T]):
        self.base = base

    def create(self, of_type: str, *args, **kwargs):
        """Create an object, instance of ``self.base``

        Parameters
        ----------
        of_type: str
            Name of class, subclass of ``self.base``. Capitalization insensitive

        args: *
            Positional arguments to construct the given class.

        kwargs: **
            Keyword arguments to construct the given class.
        """

        constructor = self.get_class(of_type)
        return constructor(*args, **kwargs)

    def get_class(self, of_type: str) -> type[T]:
        """Get the class object (not instantiated)

        Parameters
        ----------
        of_type: str
            Name of class, subclass of ``self.base``. Capitalization insensitive
        """
        of_type = of_type.lower()
        constructors = self.get_classes()

        if of_type not in constructors:
            raise NotImplementedError(
                f"Could not find implementation of {self.base.__name__}, type = '{of_type}'\n"
                "Currently, there is an implementation for types:\n"
                f"{sorted(constructors.keys())}"
            )

        return constructors[of_type]

    def get_classes(self) -> dict[str, type[T]]:
        """Get children classes of ``self.base``"""
        _import_modules(self.base)
        return get_all_types(self.base, self.base.__name__)


class Factory(ABCMeta):
    """Deprecated, will be removed in v0.3.0. See GenericFactory instead"""

    def __init__(cls, names, bases, dictionary):
        super().__init__(names, bases, dictionary)
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

        raise NotImplementedError(
            f"Could not find implementation of {cls.__base__.__name__}, type = '{of_type}'\n"
            "Currently, there is an implementation for types:\n"
            f"{sorted(cls.types.keys())}"
        )


def compute_identity(size: int = 16, **sample) -> str:
    """Compute a unique hash out of a dictionary

    Parameters
    ----------
    size: int
        size of the unique hash

    **sample:
        Dictionary to compute the hash from

    """
    sample_hash = hashlib.sha256()

    for k, v in sorted(sample.items()):
        sample_hash.update(k.encode("utf8"))

        if isinstance(v, dict):
            sample_hash.update(compute_identity(size, **v).encode("utf8"))
        else:
            sample_hash.update(str(v).encode("utf8"))

    return sample_hash.hexdigest()[:size]


# pylint: disable = unused-argument
def _handler(signum, frame):
    log.error("Oríon has been interrupted.")
    raise KeyboardInterrupt


@contextmanager
def sigterm_as_interrupt():
    """Intercept ``SIGTERM`` signals and raise ``KeyboardInterrupt`` instead"""
    # Signal only works inside the main process
    previous = signal.signal(signal.SIGTERM, _handler)

    yield None

    signal.signal(signal.SIGTERM, previous)
