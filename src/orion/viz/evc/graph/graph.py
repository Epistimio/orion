# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.evc.graph.graph` -- Build and image graph based on exp-fork tree
=======================================================================================

.. module:: vizualization
   :platform: Unix
   :synopsis: Obtains graph object and outputs an image for research purposes
              as an image or as a tex based visualization

"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph


class EvcGraph(object):
    r"""Image graph object

    Nodes are given as EVC nodes. Given these EVC nodes we would check
    if the node is the current root if we would want to create a the
    graph of the whole EVC tree. But if not given the root or a specific
    node, we would want to print out just the
    subgraph of the EVC tree.
    """

    def __init__(self, evc_node):
        """Initialize the image graph"""
        '''
        We assume that when the object get's constructred
        an EVC node is passed through.
        '''
        self.evc_node = evc_node

        self.evc_root = self.evc_node.root
        self.root_item = self.evc_root.item
        '''
        Create a networkx graph and allocate
        it in memory
        '''
        self.graph = nx.DiGraph()

        '''
        Build graph based on
        EVC nodes. Would it better be a
        property of the class?
        '''
        plt.clf()
        self.build_graph()

    def build_graph(self):
        """Convert the EVC tree to the networkx graph
        Algorithm:
            1. check if root is true
                1.1 if true call root() on evc node
                    to return root
            2. do a preordertraversal of the tree and
               build a nn.Graph()
        """
        curr_node = self.evc_node

        while curr_node.parent is not None:
            self.graph.add_node(curr_node.parent.item)
            self.graph.add_edge(curr_node.parent.item,
                                curr_node.item)
            curr_node = curr_node.parent

        '''
        Traverse the tree preorderly and then add
        EVC node to the nn.Graph() graph
        '''
        for node in self.evc_node:
            self.graph.add_node(node.item)
            for child in node.children:
                self.graph.add_edge(child.item, node.item)

    def image_visualize(self, path, title='EVC Graph'):
        """Visualize graph in a path"""
        '''
        Setup title
        '''
        plt.title(title)
        '''
        Specify hierarchical position
        '''
        pos = graphviz_layout(self.graph, prog='dot')
        '''
        Use nx.draw to draw the image
        '''

        seq = list(self.graph.nodes)

        nx.draw(self.graph, pos=pos, with_labels=True,
                color='blue', node_size=[len(node) * 300 for node in seq])
        '''
        Save the figure to a path
        '''
        plt.savefig(path)
        plt.clf()

    def image_visualize_dot(self, path):
        """Visualize image to dot format"""
        '''
        import respected pydot - write dot from networkx
        '''
        from networkx.drawing.nx_pydot import write_dot
        '''
        Adjust position for hierarchy
        '''
        pos = nx.nx_agraph.graphviz_layout(self.graph)
        '''
        draw the graph
        '''
        nx.draw(self.graph, pos=pos, with_labels=True)
        '''
        Write to file
        '''
        write_dot(self.graph, path)

    def image_visualize_custom(self, path_fname, choice='dot'):
        """Visualize graph in a path to a dot file.
        Choices for output: canon cmap cmapx cmapx_np dot dot_json eps fig v
        imap imap_np ismap json json0 mp pic plain plain-ext pov ps ps2 svg
        svgz tk vml vmlz xdot xdot1.2 xdot1.4 xdot_json'
        """
        '''
        do a draw
        '''
        dot_file = to_agraph(self.graph)
        '''
        create graphviz layout
        '''
        dot_file.layout('dot')
        '''
        write it to path
        '''
        dot_file.draw(str(path_fname) + '.' + str(choice))

    def __del__(self):
        """Destructor"""
        del self.graph


if __name__ == "__main__":
    from orion.core.evc.tree import TreeNode

    A = TreeNode("exp1")
    B = TreeNode("exp2", A)
    C = TreeNode("exp3", A)
    D = TreeNode("exp4", A)
    E = TreeNode("exp5", A)
    F = TreeNode("exp6", B)
    G = TreeNode("exp7", B)
    H = TreeNode("exp8", E)
    J = TreeNode("exp9", H)
    K = TreeNode("exp10", G)

    r'''
        # Gives this tree
        # a
        # |   \  \   \
        # b    c  d   e
        # | \         |
        # f  g        h
    '''

    '''
    a is root of EVC tree
    '''

    TEST_GRAPH = EvcGraph(A)
    # print nodes in graph
    print(list(TEST_GRAPH.graph.nodes))
    print(TEST_GRAPH.graph.adj)
    # write to disk
    TEST_GRAPH.image_visualize('./tmp/graph.png')

    '''
    a is root of EVC tree
    '''
    TEST_GRAPH2 = EvcGraph(B)

    # print nodes in graph
    print(list(TEST_GRAPH2.graph.nodes))
    print(TEST_GRAPH2.graph.adj)

    # write to disk
    TEST_GRAPH2.image_visualize('./tmp/subgraph.png')

    TEST_GRAPH3 = EvcGraph(G)

    # print nodes in graph
    print(list(TEST_GRAPH3.graph.nodes))
    print(TEST_GRAPH3.graph.adj)

    # write to disk
    TEST_GRAPH3.image_visualize('./tmp/subsubgraph.png')
