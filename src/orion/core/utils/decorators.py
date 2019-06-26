# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils.decorators` -- Custom decorators for Oríon
=================================================================

.. module:: decorators
   :platform: Unix
   :synopsis: Custom decorators.
"""


def register_check(check_list, msg):
    """Decorate a function by adding it to a list."""
    def wrap(func):
        """Wrapper."""
        check_list.append(func)
        func.msg = msg

        def wrapped_func(*args):
            """Wrap function."""
            return func(*args)

        return wrapped_func
    return wrap
