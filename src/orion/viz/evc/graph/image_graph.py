# -*- coding: utf-8 -*-
"""
:mod:`orion.viz.evc.graph.image_graph` -- Build and image graph based on exp-fork tree
==============================================================================

.. module:: vizualization
   :platform: Unix
   :synopsis: Obtains graph object and outputs an image for research purposes

"""
import networkx as nx
import matplotlib.pyplot as plt

class image_graph(object):
    r"""Image graph object
    
    Nodes are given as EVC nodes

    Given these EVC nodes we would check if the node is the current root if 
    we would want to create a the graph of the whole EVC tree. But if not
    given the root or a specific node, we would want to print out just the 
    subgraph of the EVC tree.
    
    """
    def __init__(self):
        super(image_graph, self, evc_node).__init__()
        """Initialize the image graph

        """

        '''
        We assume that when the object get's constructred 
        an EVC node is passed through.
        '''
        self.evc_node = evc_node
        
        '''
        Create a networkx graph and allocate
        it in memory
        '''
        self.graph = nx.Graph()

    def build_graph(self, root=True):
        """Convert the EVC tree to the networkx graph
        
        Algorithm:
            1. check if root is true
                1.1 if true call root() on evc node 
                    to return root
            2. do a preordertraversal of the tree and
               build a nn.Graph() 

        """

        if root:
            self.evc_node = self.evc_node.root()
        
        '''
        Traverse the tree preorderly and then add 
        EVC node to the nn.Graph() graph
        '''
        for node in self.evc_node:
            # current node is say, root
            self.graph.add_node(node)
            # its children connected with the current node
            # is connected with an edge
            self.graph.add_edge(node, node.next())
            # we've built our tree

    def visualize(self, path):
        """Visualize graph in a path
        """
        nx.draw(self.graph)
        '''
        Use nx.draw to draw the image
        '''
        plt.savefig(path)
        '''
        Save the figure to a path
        '''

    def visualize_pydot(self, path):
        """Visualize graph in a path
        to a pydot file
        """
        from networkx.drawing.nx_pydot import write_dot
        '''
        import respected pydot - write dot from networkx
        '''
        pos = nx.nx_agraph.graphviz_layout(self.graph)
        '''
        create layout
        '''
        nx.draw(G, pos=pos)
        '''
        do a draw
        '''
        write_dot(G, path)
        '''
        write it to path
        '''
