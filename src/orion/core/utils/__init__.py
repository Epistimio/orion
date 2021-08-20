# -*- coding: utf-8 -*-
"""
Package-wide useful routines
============================

"""

import logging
import os
from abc import ABCMeta
from collections import defaultdict
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
    base = import_module(cls.__base__.__module__)
    try:
        py_files = glob(os.path.abspath(os.path.join(base.__path__[0], "[A-Za-z]*.py")))
        py_mods = map(
            lambda x: "." + os.path.split(os.path.splitext(x)[0])[1], py_files
        )
        for py_mod in py_mods:
            cls.modules.append(import_module(py_mod, package=cls.__base__.__module__))
    except AttributeError:
        # This means that base class and implementations reside in a module
        # itself and not a subpackage.
        pass

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


class Factory(ABCMeta):
    """Instantiate appropriate wrapper for the infrastructure based on input
    argument, ``of_type``.

    Attributes
    ----------
    types : dict of subclasses of ``cls.__base__``
       Updated to contain all possible implementations currently. Check out code.

    """

    def __init__(cls, names, bases, dictionary):
        """Search in directory for attribute names subclassing `bases[0]`"""
        super(Factory, cls).__init__(names, bases, dictionary)
        cls.types = {}
        try:
            _import_modules(cls)
        except ImportError:
            pass
        _set_typenames(cls)

    def __call__(cls, of_type, *args, **kwargs):
        """Create an object, instance of ``cls.__base__``, on first call.

        :param of_type: Name of class, subclass of ``cls.__base__``, wrapper
           of a database framework that will be instantiated on the first call.
        :param args: positional arguments to initialize ``cls.__base__``'s instance (if any)
        :param kwargs: keyword arguments to initialize ``cls.__base__``'s instance (if any)

        .. seealso::
           `Factory.types` keys for values of argument `of_type`.

        .. seealso::
           Attributes of ``cls.__base__`` and ``cls.__base__.__init__`` for
           values of `args` and `kwargs`.

        .. note:: New object is saved as `Factory`'s internal state.

        :return: The object which was created on the first call.
        """
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
