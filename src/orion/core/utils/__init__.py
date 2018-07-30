# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils` -- Package-wide useful routines
=======================================================

.. module:: utils
   :platform: Unix
   :synopsis: Helper functions useful in possibly all :mod:`orion.core`'s modules.
"""

from abc import ABCMeta
from glob import glob
from importlib import import_module
import logging
import os

import pkg_resources


log = logging.getLogger(__name__)


def get_qualified_name(package, name):
    return package + '.' + name


class SingletonError(ValueError):
    """Exception to be raised when someone provides arguments to build
    an object from a already-instantiated `SingletonType` class.
    """

    def __init__(self):
        """Pass the same constant message to ValueError underneath."""
        super().__init__("A singleton instance has already been instantiated.")


class SingletonType(type):
    """Metaclass that implements the singleton pattern for a Python class."""

    def __init__(cls, name, bases, dictionary):
        """Create a class instance variable and initiate it to None object."""
        super(SingletonType, cls).__init__(name, bases, dictionary)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        """Create an object if does not already exist, otherwise return what there is."""
        if cls.instance is None:
            cls.instance = super(SingletonType, cls).__call__(*args, **kwargs)
        elif args or kwargs:
            raise ValueError("A singleton instance has already been instantiated.")
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
            py_files = glob(os.path.abspath(os.path.join(base.__path__[0] + '/**/',
                                                         '[A-Za-z]*.py')), recursive=True)
            py_mods = map(lambda x: x.split('orion/src/')[1].replace('/', '.')[:-3], py_files)
            for py_mod in py_mods:
                cls.modules.append(import_module(py_mod))
        except AttributeError:
            # This means that base class and implementations reside in a module
            # itself and not a subpackage.
            pass

        # Get types advertised through entry points!
        for entry_point in pkg_resources.iter_entry_points(cls.__base__.__name__):
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
        cls.typenames = list(map(lambda x: get_qualified_name(x.__module__,
                                                              x.__name__).lower(), cls.types))
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
        module, name = of_type
        qualified_name = get_qualified_name(module, name).lower()

        for inherited_class in cls.types:
            inh_qualified_name = get_qualified_name(inherited_class.__module__,
                                                    inherited_class.__name__).lower()
            if inh_qualified_name == qualified_name:
                return inherited_class.__call__(*args, **kwargs)

        error = "Could not find implementation of {0}, type = '{1}'".format(
            cls.__base__.__name__, qualified_name)
        error += "\nCurrently, there is an implementation for types:\n"
        error += str(cls.typenames)
        raise NotImplementedError(error)


class SingletonFactory(AbstractSingletonType, Factory):
    """Wrapping `Factory` with `SingletonType`. Keep compatibility with `AbstractSingletonType`."""

    pass


class Concept(object):
    """Define an abstract concept. Be it an algorithm, a plotter or whatever."""
    def __init__(self, *args, **kwargs):
        """Create a specifig instance of said concept"""

        # Get base class information
        base_class = type(self).__base__
        name = base_class.name
        module = getattr(base_class, "implementation_module", self.__module__)
        module = getattr(self, "implementation_module", module)
        module_addendum = getattr(base_class, 'module_addendum', '')

        # Create a factory instance for the base class
        self.factory = Factory('Factory', (base_class,), globals())

        log.debug("Creating %s object of %s type with parameters:\n%s",
                  name, type(self).__name__, kwargs)

        if module_addendum != '' and not module.endswith(module_addendum):
            module += '.' + module_addendum

        for varname, param in kwargs.items():
            if isinstance(param, dict) and len(param) > 0:
                try:
                    sub_type = list(param)[0]
                    sub_kwargs = param[sub_type]

                    if isinstance(sub_kwargs, dict):
                        qualified_name = get_qualified_name(module, sub_type)

                        param = self.factory((qualified_name, sub_type),
                                             *args, **sub_kwargs)

                    if isinstance(param, dict) and len(param) > 1:
                        for subvar, subparam, in param.items()[1:]:
                            setattr(self, subvar, subparam)

                except NotImplementedError:
                    pass

            elif isinstance(param, str) and \
                    get_qualified_name(get_qualified_name(module, param), param) \
                    in self.factory.typenames:
                param = self.factory((get_qualified_name(module, param), param), *args)

            setattr(self, varname, param)
