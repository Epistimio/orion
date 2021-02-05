# -*- coding: utf-8 -*-
"""
Tree data structure for the experiment version control system
=============================================================

Tree data structure for the experiment version control system

Experiment version control requires building trees of the experiments so
that we can fetch trials from one experiment to another or navigate from
one experiment to another to visualise different statistics.

TreeNode and tree iterators support the tree data structure of the experiment version control.
TreeNode is a generic class which can carry arbitrary python objects. It comes with basic methods to
set parent and children. A method `map` allows to apply functions recursively on the tree in a
generic manner.

"""


# pylint: disable=too-few-public-methods
class PreOrderTraversal(object):
    """Iterate on a tree in a pre-order traversal fashion

    Attributes
    ----------
    stack: list of `orion.core.evc.tree.TreeNode`
        Nodes logged during iteration

    """

    __slots__ = ("stack",)

    def __init__(self, tree_node):
        """Initialize the stack for iteration"""
        self.stack = [tree_node]

    def __iter__(self):
        """Get the iterator"""
        return self

    def __next__(self):
        """Get the next node in pre-order traversal"""
        try:
            node = self.stack.pop()
        except IndexError:
            raise StopIteration

        self.stack += node.children[::-1]

        return node


# pylint: disable=too-few-public-methods
class DepthFirstTraversal(object):
    """Iterate on a tree in a pre-order traversal fashion

    Attributes
    ----------
    stack: list of `orion.core.evc.tree.TreeNode`
        Nodes logged during iteration
    seen: set of `orion.core.evc.tree.TreeNode`
        Nodes which have been returned during iteration

    """

    __slots__ = ("stack", "seen")

    def __init__(self, tree_node):
        """Initialize the stack and set of seen nodes for iteration"""
        self.stack = [tree_node]
        self.seen = set()

    def _compute_potential(self):
        """Filter out seen nodes from the stack"""
        if not self.stack:
            return []

        return list(filter(lambda n: n not in self.seen, self.stack[-1].children))

    def __iter__(self):
        """Get the iterator"""
        return self

    def __next__(self):
        """Get the next node in depth-first traversal"""
        potential = self._compute_potential()
        while self.stack and potential:
            self.stack += potential[::-1]
            potential = self._compute_potential()

        try:
            node = self.stack.pop()
        except IndexError:
            raise StopIteration

        self.seen.add(node)

        return node


class TreeNode(object):
    r"""Tree node data structure

    Nodes have an attribute item to carry arbitrary information. A node may only have one parent and
    can have as many children as desired.

    Parents can be set at initialiation or via `node.set_parent`. Setting a parent automatically set
    the current node as the child of the parent.

    Children can be set at initialiation or via `node.add_children`. Setting children automatically
    set their parent as the current node.

    Tree of nodes are iterable, by default with preorder traversal.

    .. seealso::
        `orion.core.evc.tree.PreOrderTraversal`
        `orion.core.evc.tree.DepthFirstTraversal`

    Attributes
    ----------
    item: object
        Can be anything
    parent: None or instance of `orion.core.evc.tree.TreeNode`
        The parent of the current node, None if the current node is the root.
    children: None or list of instances of `orion.core.evc.tree.TreeNode`
        The children of the curent node.
    root: instance of `orion.core.evc.tree.TreeNode`
        The top node of the current tree. The root node returns itself.

    Examples
    --------
    .. code-block:: python
        :linenos:

        a = TreeNode("a")
        b = TreeNode("b", a)
        c = TreeNode("c", a)
        d = TreeNode("d", a)
        e = TreeNode("e", a)

        f = TreeNode("f", b)
        g = TreeNode("g", b)

        h = TreeNode("h", e)

        # Gives this tree
        # a
        # |   \  \   \
        # b    c  d   e
        # | \         |
        # f  g        h

        g.set_parent(e)

        # Gives this tree
        # a
        # |  \  \   \
        # b   c  d   e
        # |          | \
        # f          h g

        c.add_children(h, g)

        # Gives this tree
        # a
        # |  \    \   \
        # b   c    d   e
        # |   | \
        # f   h  g

        a.drop_children(c)

        # Gives this tree
        # a
        # |  \   \
        # b   d   e
        # |
        # f

    """

    __slots__ = ("_item", "_parent", "_children")

    def __init__(self, item, parent=None, children=tuple()):
        """Initialize node with item, parent and children

        .. seealso::
            :class:`orion.core.evc.tree.TreeNode` for information about the attributes
        """
        self._item = item
        self._parent = None
        self._children = []

        if parent is not None:
            self.set_parent(parent)
        if children:
            self.add_children(*children)

    @property
    def item(self):
        """Get item of the node which may contain arbitrary objects"""
        return self._item

    @item.setter
    def item(self, new_item):
        """Set item of the node with arbitrary objects"""
        self._item = new_item

    @property
    def parent(self):
        """Get parent of the node, None if no parent"""
        return self._parent

    def drop_parent(self):
        """Drop the parent of the node, do nothing if no parent

        Note that the node will be removed from the children of the parent as well
        """
        if self.parent is not None:
            self._parent.drop_children(self)

    def set_parent(self, node):
        """Set the parent of the node

        Note that setting a new parent will have the effect of dropping the previous parent, hence
        dropping this current node from the previous parent's children list.

        .. seealso::
            `orion.core.evc.tree.TreeNode.drop_parent`
        """
        if node is self.parent:
            return

        if node is not None and not isinstance(node, TreeNode):
            raise TypeError("Cannot set parent to %s" % str(node))

        if self.parent is not None:
            self.drop_parent()

        if node is not None:
            node.add_children(self)

    @property
    def children(self):
        """Get children of the node, empty list if no children"""
        return self._children

    def drop_children(self, *nodes):
        """Drop the children of the node, do nothing if no parent

        If no nodes are passed, the method will drop all the children of the current node.

        Note that the parent of the given node will be removed as well

        Raises
        ------
        ValueError
            If one of the given nodes is not a children of the current node.

        """
        if not nodes:
            nodes = self.children

        for child in list(nodes):
            del self._children[self._children.index(child)]
            # pylint: disable=protected-access
            child._parent = None

    def add_children(self, *nodes):
        """Add children to the current node

        Note that added children will have their parent set to the current node as well.

        .. seealso::
            `orion.core.evc.tree.TreeNode.drop_children`
        """
        for child in nodes:
            if child is not None and not isinstance(child, TreeNode):
                raise TypeError("Cannot add %s to children" % str(child))

            if child not in self._children:
                # TreeNode.set_parent uses add_children so using it here could cause an infinit
                # recursion. add_children() gets the dirty job done.
                child.drop_parent()
                # pylint: disable=protected-access
                child._parent = self
                self._children.append(child)

    @property
    def root(self):
        """Get the root of the tree

        Root node returns itself
        """
        if self.parent is None:
            return self

        return self.parent.root

    def map(self, function, node):
        r"""Apply a function recursively on the tree

        The function can be applied upwards on parents or downwards on children. The direction is
        defined by passing self.parent or self.children as the node argument.

        Parameters
        ----------
        function : callable
            Callable object to which will be passed the current node plus the parent node or the
            children nodes, depending on the direction of function application.
            If map on parents, callable(self, rval_parent_node)
            If map on children, callable(self, rval_children_nodes).
            Note that the callable object is expected to return an object which will be set as the
            current node's item (in the resulting tree), and the parent node or the children nodes
            depending on the direction of function application.
        node: None, `orion.core.evc.TreeNode` or list
            Can be either
            `None`: function is applied on current node only
            `self.parent`: function is applied recursively climbing up the tree until the root
            `self.children`: function is applied recursively going down the tree until the leafs

        Examples
        --------
        .. code-block:: python
            :linenos:

            # Tree structure
            # a
            # |   \  \   \
            # b    c  d   e
            # | \         |
            # f  g        h
            #

            root = TreeNode(1)
            b = TreeNode(1, root)
            TreeNode(1, root)
            TreeNode(1, root)
            e = TreeNode(1, root)

            f = TreeNode(1, b)
            TreeNode(1, b)

            h = TreeNode(1, e)

            def increment(node, children):
                for child in TreeNode(0, None, children):
                    child.item += 1

                return node.item + 1, children

            # Should return
            #
            # 2
            # |   \  \   \
            # 3    3  3   3
            # | \         |
            # 4  4        4

            rval = root.map(increment, root.children)
            assert [node.item for node in rval] == [2, 3, 4, 4, 3, 3, 3, 4]

            def increment_parent(node, parent):
                if parent is not None:
                    for parent in parent.root:
                        parent.item += 1

                return node.item + 1, parent

            # Should return
            #
            # 4
            # |
            # 3
            # |
            # 2

            rval = f.map(increment_parent, f.parent)
            assert [node.item for node in rval.root] == [4, 3, 2]

            rval = h.map(increment_parent, h.parent)
            assert [node.item for node in rval.root] == [4, 3, 2]

        """
        if node is None:
            rval, _ = function(self, None)
            return TreeNode(rval)
        elif node is self.parent:
            rval_parent_node = node.map(function, node.parent)
            rval, parent_node = function(self, rval_parent_node)
            return TreeNode(rval, parent_node)
        elif node is self.children:
            rval_children_nodes = [
                child.map(function, child.children) for child in self.children
            ]
            rval, children_nodes = function(self, rval_children_nodes)
            return TreeNode(rval, parent=None, children=children_nodes)
        else:
            raise ValueError("Invalid nodes: %s" % str(node))

    def __iter__(self):
        """Iterate on the tree with pre-order traversal"""
        return PreOrderTraversal(self)

    def __repr__(self):
        """Represent the object as a string."""
        parent = self.parent
        if parent is not None:
            parent = parent.item

        children = [child.item for child in self.children]

        return "%s(%s, parent=%s, children=%s)" % (
            self.__class__.__name__,
            str(self.item),
            str(parent),
            str(children),
        )


def flattened(trials_tree):
    """Get a list of the tree items in pre-order traversal"""
    return [node.item for node in trials_tree]
