# -*- coding: utf-8 -*-
"""
Utility functions for importing modules
=======================================

Conversion functions between various data types used in framework's ecosystem.

"""

import os


def load_modules_in_path(path, filter_function=None):
    """
    Load all modules inside `path` and return a list of those
    fitting the filter function.
    """
    this_module = __import__(path, fromlist=[""])
    file_path = this_module.__path__[0]

    files = list(
        map(
            lambda f: f.split(".")[0],
            filter(lambda f2: f2.endswith("py"), os.listdir(file_path)),
        )
    )

    modules = map(lambda f: __import__(path + "." + f, fromlist=[""]), files)

    if filter_function is not None:
        modules = filter(filter_function, modules)

    return list(modules)
