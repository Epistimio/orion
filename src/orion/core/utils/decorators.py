# -*- coding: utf-8 -*-


def register_check(check_list, msg):
    """Decorate a function by adding it to a list."""
    def wrap(func):
        check_list.append(func)
        func.msg = msg

        def wrapped_func(*args):
            return func(*args)

        return wrapped_func
    return wrap
