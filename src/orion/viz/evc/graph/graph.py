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
        if self.evc_node != self.evc_node.root:
            '''Check if accidently we
                passed a root, if yes root do nothing
            '''
            self.graph.add_node(self.evc_node.parent.item,
                                color='blue', size=500)
            self.graph.add_edge(self.evc_node.parent.item, self.evc_node.item,
                                color='red', weight=0.84, size=300)

        '''
        Traverse the tree preorderly and then add
        EVC node to the nn.Graph() graph
        '''
        for node in self.evc_node:
            self.graph.add_node(node.item,
                                color='blue', size=500)
            for child in node.children:
                self.graph.add_edge(node.item, child.item, color='red',
                                    weight=0.84, size=300)

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
        nx.draw(self.graph, pos=pos, with_labels=True)
        '''
        Save the figure to a path
        '''
        plt.savefig(path)

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

    A = TreeNode("a")
    B = TreeNode("b", A)
    C = TreeNode("c", A)
    D = TreeNode("d", A)
    E = TreeNode("e", A)
    F = TreeNode("f", B)
    G = TreeNode("g", B)
    H = TreeNode("h", E)

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

    # write to disk
    TEST_GRAPH.image_visualize('./tmp/graph.png')
    TEST_GRAPH.image_visualize_dot('./tmp/graph.dot')

    '''
    a is root of EVC tree
    '''

    TEST_GRAPH2 = EvcGraph(B)

    # print nodes in graph
    print(list(TEST_GRAPH2.graph.nodes))

    # write to disk
    TEST_GRAPH2.image_visualize('./tmp/subgraph.png')
    TEST_GRAPH2.image_visualize_dot('./tmp/subgraph.dot')
