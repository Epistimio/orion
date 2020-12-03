"""Test for generic :class:`orion.core.evc.tree`"""

from orion.core.evc.tree import (
    DepthFirstTraversal,
    PreOrderTraversal,
    TreeNode,
    flattened,
)


def test_node_creation():
    """Test empty initialization of tree node"""
    TreeNode("test")
    TreeNode("test", None, tuple())


def test_node_creation_with_parent():
    """Test assignement of parent on initialization"""
    parent = TreeNode("test")
    child = TreeNode("test", parent)

    assert child.parent is parent

    assert parent.children[0] is child


def test_node_set_parent():
    """Test assignement of parent"""
    parent = TreeNode("test")
    child = TreeNode("test")

    child.set_parent(parent)

    assert child.parent is parent

    assert parent.children[0] is child


def test_node_set_no_parent():
    """Test assignement of no parent, meaning droping the parent"""
    parent = TreeNode("test")
    child = TreeNode("test")

    child.set_parent(parent)

    assert child.parent is parent
    assert parent.children[0] is child

    child.set_parent(None)

    assert child.parent is None
    assert len(parent.children) == 0


def test_node_drop_parent():
    """Test dropping parent"""
    parent = TreeNode("test")
    child = TreeNode("test")

    child.set_parent(parent)

    assert child.parent is parent
    assert parent.children[0] is child

    child.drop_parent()

    assert child.parent is None
    assert len(parent.children) == 0


def test_node_add_child():
    """Test assignement of child"""
    parent = TreeNode("test")
    child = TreeNode("test")

    parent.add_children(child)

    assert parent.children[0] is child
    assert child.parent is parent


def test_node_remove_child():
    """Test removing child"""
    parent = TreeNode("test")
    child = TreeNode("test")

    parent.add_children(child)

    assert child.parent is parent
    assert parent.children[0] is child

    parent.drop_children(child)

    assert len(parent.children) == 0
    assert child.parent is None


def test_node_add_children():
    """Test assignement of children"""
    parent = TreeNode("test")
    child1 = TreeNode("test1")
    child2 = TreeNode("test2")

    parent.add_children(child1, child2)

    assert parent.children[0] is child1
    assert parent.children[1] is child2
    assert child1.parent is parent
    assert child2.parent is parent


def test_node_remove_children_sequentially():
    """Test removing children one after the other"""
    parent = TreeNode("test")
    child1 = TreeNode("test1")
    child2 = TreeNode("test2")

    parent.add_children(child1, child2)

    assert parent.children[0] is child1
    assert parent.children[1] is child2
    assert child1.parent is parent
    assert child2.parent is parent

    parent.drop_children(child1)

    assert len(parent.children) == 1
    assert child1.parent is None
    assert parent.children[0] is child2
    assert child2.parent is parent

    parent.drop_children(child2)

    assert len(parent.children) == 0
    assert child1.parent is None
    assert child2.parent is None


def test_node_remove_children_in_batch():
    """Test removing children all at the same time"""
    parent = TreeNode("test")
    child1 = TreeNode("test1")
    child2 = TreeNode("test2")

    parent.add_children(child1, child2)

    assert parent.children[0] is child1
    assert parent.children[1] is child2
    assert child1.parent is parent
    assert child2.parent is parent

    parent.drop_children(child1, child2)

    assert len(parent.children) == 0
    assert child1.parent is None
    assert child2.parent is None


def test_node_remove_all_children():
    """Test drop_children() drops all of them"""
    parent = TreeNode("test")
    child1 = TreeNode("test1")
    child2 = TreeNode("test2")

    parent.add_children(child1, child2)

    assert parent.children[0] is child1
    assert parent.children[1] is child2
    assert child1.parent is parent
    assert child2.parent is parent

    parent.drop_children()

    assert len(parent.children) == 0
    assert child1.parent is None
    assert child2.parent is None


def test_parent_parent():
    """Test path through two level of parents"""
    grand_parent = TreeNode("test")
    parent = TreeNode("test", grand_parent)
    child = TreeNode("test", parent)

    assert child.parent is parent
    assert child.parent.parent is grand_parent
    assert grand_parent.children[0] is parent
    assert grand_parent.children[0].children[0] is child


def test_children_children():
    """Test path through two level of children"""
    child = TreeNode("test")
    parent = TreeNode("test", children=[child])
    grand_parent = TreeNode("test", children=[parent])

    assert child.parent is parent
    assert child.parent.parent is grand_parent
    assert grand_parent.children[0] is parent
    assert grand_parent.children[0].children[0] is child


def test_root():
    """Test root access from leaf"""
    child = TreeNode("test")
    parent = TreeNode("test", children=[child])
    grand_parent = TreeNode("test", children=[parent])

    assert child.root is grand_parent
    assert child.root is parent.root
    assert grand_parent.root is grand_parent


def test_node_parent_reinsertion():
    """Test that replacing a parent with the same one does not change anything"""
    parent = TreeNode("test")
    child1 = TreeNode("test", parent)
    child2 = TreeNode("test", parent)

    assert child1.parent is parent
    assert child2.parent is parent
    assert parent.children[0] is child1
    assert parent.children[1] is child2

    child1.set_parent(parent)

    assert child1.parent is parent
    assert child2.parent is parent
    assert parent.children[0] is child1
    assert parent.children[1] is child2

    child2.set_parent(parent)

    assert child1.parent is parent
    assert child2.parent is parent
    assert parent.children[0] is child1
    assert parent.children[1] is child2


def test_node_children_reinsertion():
    """Test that replacing a children with the same one does not change anything"""
    parent = TreeNode("test")
    child1 = TreeNode("test", parent)
    child2 = TreeNode("test", parent)

    assert child1.parent is parent
    assert child2.parent is parent
    assert parent.children[0] is child1
    assert parent.children[1] is child2

    parent.add_children(child1)

    assert child1.parent is parent
    assert child2.parent is parent
    assert parent.children[0] is child1
    assert parent.children[1] is child2

    parent.add_children(child2)

    assert child1.parent is parent
    assert child2.parent is parent
    assert parent.children[0] is child1
    assert parent.children[1] is child2


def test_node_set_parent_drop_children():
    """Test that replacing a parent removes the node from parent's children"""
    grand_parent = TreeNode("test")
    parent = TreeNode("test", grand_parent)
    child = TreeNode("test", parent)

    assert child.parent is parent
    assert child.parent.parent is grand_parent
    assert grand_parent.children[0] is parent
    assert grand_parent.children[0].children[0] is child

    # That's silly...
    child.set_parent(grand_parent)

    assert child.parent is grand_parent
    assert len(grand_parent.children) == 2
    assert grand_parent.parent is None
    assert len(parent.children) == 0
    assert parent.parent is grand_parent


def test_node_add_children_drop_children():
    """Test that replacing a children removes the node from previous parent's children"""
    grand_parent = TreeNode("test")
    parent = TreeNode("test", grand_parent)
    child = TreeNode("test", parent)

    assert child.parent is parent
    assert child.parent.parent is grand_parent
    assert grand_parent.children[0] is parent
    assert grand_parent.children[0].children[0] is child

    # That's silly...
    grand_parent.add_children(child)

    assert child.parent is grand_parent
    assert len(grand_parent.children) == 2
    assert grand_parent.parent is None
    assert len(parent.children) == 0
    assert parent.parent is grand_parent


def test_preorder_traversal():
    """Test order retrieval of DepthFirstTraversal"""
    # a
    # |   \  \   \
    # b    c  d   e
    # | \         |
    # f  g        h
    root = TreeNode("a")
    b = TreeNode("b", root)
    TreeNode("c", root)
    TreeNode("d", root)
    e = TreeNode("e", root)

    TreeNode("f", b)
    TreeNode("g", b)

    TreeNode("h", e)

    rval = [node.item for node in PreOrderTraversal(root)]
    assert rval == ["a", "b", "f", "g", "c", "d", "e", "h"]


def test_depth_first_traversal():
    """Test order retrieval of DepthFirstTraversal"""
    # a
    # |   \  \   \
    # b    c  d   e
    # | \         |
    # f  g        h
    root = TreeNode("a")
    b = TreeNode("b", root)
    TreeNode("c", root)
    TreeNode("d", root)
    e = TreeNode("e", root)

    TreeNode("f", b)
    TreeNode("g", b)

    TreeNode("h", e)

    rval = [node.item for node in DepthFirstTraversal(root)]
    assert rval == ["f", "g", "b", "c", "d", "h", "e", "a"]


def test_map_children():
    """Test recursive application of map on children"""
    # a
    # |   \  \   \
    # b    c  d   e
    # | \         |
    # f  g        h
    #
    # Should become
    #
    # 2
    # |   \  \   \
    # 3    3  3   3
    # | \         |
    # 4  4        4

    root = TreeNode(1)
    b = TreeNode(1, root)
    TreeNode(1, root)
    TreeNode(1, root)
    e = TreeNode(1, root)

    TreeNode(1, b)
    TreeNode(1, b)

    TreeNode(1, e)

    def increment(node, children):
        for child in TreeNode(0, None, children):
            child.item += 1

        return node.item + 1, children

    rval = root.map(increment, root.children)
    assert [node.item for node in rval] == [2, 3, 4, 4, 3, 3, 3, 4]


def test_map_parent():
    """Test recursive application of map on parents"""
    # a
    # |   \  \   \
    # b    c  d   e
    # | \         |
    # f  g        h
    #
    # Will give
    #
    # 4
    # |
    # 3
    # |
    # 2

    root = TreeNode(1)
    b = TreeNode(1, root)
    TreeNode(1, root)
    TreeNode(1, root)
    e = TreeNode(1, root)

    f = TreeNode(1, b)
    TreeNode(1, b)

    h = TreeNode(1, e)

    def increment_parent(node, parent):
        if parent is not None:
            for parent in parent.root:
                parent.item += 1

        return node.item + 1, parent

    rval = f.map(increment_parent, f.parent)
    assert [node.item for node in rval.root] == [4, 3, 2]

    rval = h.map(increment_parent, h.parent)
    assert [node.item for node in rval.root] == [4, 3, 2]


def test_flattened():
    """Test flattened tree into a list, retrieving items"""
    # a
    # |   \  \   \
    # b    c  d   e
    # | \         |
    # f  g        h
    root = TreeNode("a")
    b = TreeNode("b", root)
    TreeNode("c", root)
    TreeNode("d", root)
    e = TreeNode("e", root)

    TreeNode("f", b)
    TreeNode("g", b)

    TreeNode("h", e)

    assert flattened(root) == ["a", "b", "f", "g", "c", "d", "e", "h"]
