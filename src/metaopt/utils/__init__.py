# -*- coding: utf-8 -*-
"""
:mod:`metaopt.utils` -- Package-wide useful routines
====================================================

.. module:: utils
   :platform: Unix
   :synopsis: Helper functions useful in possibly all :mod:`metaopt`'s modules.
"""

from abc import ABCMeta


class SingletonType(type):
    """Metaclass that implements the singleton pattern for a Python class."""

    def __init__(cls, name, bases, dictionary):
        """Create a class instance variable and initiate it to None object."""
        super(SingletonType, cls).__init__(name, bases, dictionary)
        cls.instance = None

    def __call__(cls, *args, **kwds):
        """Create an object if does not already exist, otherwise return what there is."""
        if cls.instance is None:
            cls.args = args
            cls.kwds = kwds
            cls.instance = super(SingletonType, cls).__call__(*args, **kwds)
        return cls.instance


class AbstractSingletonType(SingletonType, ABCMeta):
    """This will create singleton base classes, that need to be subclassed and implemented."""

    pass
