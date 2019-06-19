# -*- coding: utf-8 -*-


def register_check(checklist):
    """Decorate a function by adding it to a list."""
    def wrap(func):
        checklist.append(func)

        def wrapped_func(*args):
            func(*args)

        return wrapped_func
    return wrap
