# -*- coding: utf-8 -*-
"""
Utilitary functions for diffs
=============================

Utilitary functions to provided colored diffs of (multi-line) strings

"""

from difflib import Differ


def red(string):
    """Return the string wrapped in code to color it in red in user's terminal"""
    return color(string, "31m")


def green(string):
    """Return the string wrapped in code to color it in green in user's terminal"""
    return color(string, "32m")


def color(string, color_code):
    """Return the string wrapped in code to color it in given `color_code` in user's terminal"""
    return "\x1b[0;" + color_code + string + "\x1b[0m"


def colored_diff(string_a, string_b):
    """Return a multi-line colored diff of given strings `a` and `b`"""
    string_a = string_a.splitlines(keepends=True)
    string_b = string_b.splitlines(keepends=True)
    result = list(Differ().compare(string_a, string_b))

    colored_result = []
    for i, line in enumerate(result):
        line = line.strip("\n")
        if line[0] == "-" or (line[0] == "?" and result[i - 1][0] == "-"):
            line = red(line)
        elif line[0] == "+" or (line[0] == "?" and result[i - 1][0] == "+"):
            line = green(line)

        colored_result.append(line)

    return "\n".join(colored_result)
