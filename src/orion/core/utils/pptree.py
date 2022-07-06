# pylint: disable-all
"""
Utilitary functions for printing trees
======================================

Clement Michard (c) 2015

https://github.com/clemtoy/pptree

MIT License

Copyright (c) 2017 Clément Michard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

        if parent:
            self.parent.children.append(self)


def print_tree(
    current_node, childattr="children", nameattr="name", indent="", last="updown"
):
    if hasattr(current_node, nameattr):
        name = lambda node: getattr(node, nameattr)
    else:
        name = lambda node: str(node)

    children = lambda node: getattr(node, childattr)
    nb_children = lambda node: sum(nb_children(child) for child in children(node)) + 1
    size_branch = {child: nb_children(child) for child in children(current_node)}

    """ Creation of balanced lists for "up" branch and "down" branch. """
    up = sorted(children(current_node), key=lambda node: nb_children(node))
    down = []
    while up and sum(size_branch[node] for node in down) < sum(
        size_branch[node] for node in up
    ):
        down.append(up.pop())

    """ Printing of "up" branch. """
    for child in up:
        next_last = "up" if up.index(child) == 0 else ""
        next_indent = "{}{}{}".format(
            indent, " " if "up" in last else "│", " " * len(name(current_node))
        )
        print_tree(child, childattr, nameattr, next_indent, next_last)

    """ Printing of current node. """
    if last == "up":
        start_shape = "┌"
    elif last == "down":
        start_shape = "└"
    elif last == "updown":
        start_shape = " "
    else:
        start_shape = "├"

    if up:
        end_shape = "┤"
    elif down:
        end_shape = "┐"
    else:
        end_shape = ""

    print(f"{indent}{start_shape}{name(current_node)}{end_shape}")

    """ Printing of "down" branch. """
    for child in down:
        next_last = "down" if down.index(child) is len(down) - 1 else ""
        next_indent = "{}{}{}".format(
            indent, " " if "down" in last else "│", " " * len(name(current_node))
        )
        print_tree(child, childattr, nameattr, next_indent, next_last)
