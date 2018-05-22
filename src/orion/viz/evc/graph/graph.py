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

import networkx as nx
import matplotlib.pyplot as plt

class evc_graph(object):
    r"""Image graph object
    
    Nodes are given as EVC nodes

    Given these EVC nodes we would check if the node is the current root if 
    we would want to create a the graph of the whole EVC tree. But if not
    given the root or a specific node, we would want to print out just the 
    subgraph of the EVC tree.
    
    """
    def __init__(self):
        super(evc_graph, self, evc_node).__init__()
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

        '''
        Build graph based on
        EVC nodes. Would it better be a 
        property of the class?
        '''
        self.build_graph(root=True)

    def add_node(self, node, children):
        self.graph.add_node(node)
        for child in children:
            self.graph.add_edge(node, child)
        return node, children

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

        self.evc_node.map(self.add_node, self.evc_node.children)
    
    def dumps_tikz(self, layout='layered', use_label=True):
        """Return TikZ code as `str`given a graph
        
        """
        '''
        adjusting layout
        '''
        if layout not in ('layered', 'spring'):
            raise ValueError('Unknown layout: {s}'.format(s=layout))
        '''
        start at \0 string
        '''
        s = ''
        for n, d in self.graph.nodes_iter(data=True):
            # label
            label = self.graph.node[n].get('label', '')
            label = 'as={' + label + '}' if label else ''
            # geometry
            color = d.get('color', '')
            fill = d.get('fill', '')
            shape = d.get('shape', '')
            # style
            style = ', '.join(filter(None, [label, color, fill, shape]))
            style = '[' + style + ']' if style else ''
            # pack them
            s += '{n}{style};\n'.format(n=n, style=style)

        s += '\n'
        if nx.is_directed(self.graph):
            line = ' -> '
        else:
            line = ' -- '
        for u, v, d in self.graph.edges_iter(data=True):
            if use_label:
                label = d.get('label', '')
                color = d.get('color', '')
            else:
                label = str(d)
                color = ''
            if label:
                label = '"' + label + '"\' above'
            loop = 'loop' if u is v else ''
            style = ', '.join(filter(None, [label, color, loop]))
            style = ' [' + style + '] ' if style else ''
            s += str(u) + line + style + str(v) + ';\n'
        tikzpicture = (
            r'\begin{{tikzpicture}}\n'
            '\graph[{layout} layout, sibling distance=5.0cm,'
            # 'edge quotes mid,'
            'edges={{nodes={{ sloped, inner sep=10pt }} }},'
            'nodes={{circle, draw}} ]{{\n'
            '{s}'
            '}};\n'
            '\end{{tikzpicture}}\n').format(
                layout=layout,
                s=s)
        return tikzpicture

    def preamble(self, layout='layered'):
        """Return preamble and begin/end document.
        
        """
        if layout == 'layered':
            layout_lib = 'layered'
        elif layout == 'spring':
            layout_lib = 'force'
        else:
            raise ValueError(
                'Unknown which library contains layout: {s}'
                        .format(s=layout))
        document = (
            '\\n'
            '\documentclass{{standalone}}'
            '\n'
            '\usepackage{{amsmath}}\n'
            '\usepackage{{tikz}}\n'
            '\usetikzlibrary{{graphs,graphs.standard,'
            'graphdrawing,quotes,shapes}}\n'
            '\usegdlibrary{{ {layout_lib} }}\n').format(
                layout_lib=layout_lib)

        return document

    def document(self, layout, use_label):
        """Return `str` that contains a preamble and tikzpicture.
        
        """
        tikz = self.dumps_tikz(self.graph, layout, use_label=use_label)
        preamble = self.preamble(layout)
        return (
            '{preamble}\n'
            r'\begin{{document}}' '\n'
            '\n'
            '{tikz}'
            '\end{{document}}\n').format(
                preamble=preamble,
                tikz=tikz)


    def tikz_visualize(self, path):
        """Write TikZ picture as TeX file.

        """
        s = self.dumps_tikz(self.graph)
        with open(path, 'w') as f:
            f.write(s)


    def tex_visualize(self, path, use_label=True):
        """Write TeX document (use this as an example).
        
        """
        s = self.document(self.graph, layout='layered', 
                            use_label=use_label)
        with open(path, 'w') as f:
            f.write(s)

    def image_visualize(self, path):
        """Visualize graph in a path

        """
        '''
        Use nx.draw to draw the image
        '''
        nx.draw(self.graph)
        '''
        Save the figure to a path
        '''
        plt.savefig(path)

    def image_visualize_pydot(self, path):
        """Visualize graph in a path to a pydot file

        """
        '''
        import respected pydot - write dot from networkx
        '''
        from networkx.drawing.nx_pydot import write_dot
        '''
        create graphviz layout
        '''
        pos = nx.nx_agraph.graphviz_layout(self.graph)
        '''
        do a draw
        '''
        nx.draw(self.graph, pos=pos)
        '''
        write it to path
        '''
        write_dot(self.graph, path)

if __name__=="__main__":

    a = TreeNode("a")
    b = TreeNode("b", a)
    c = TreeNode("c", a)
    d = TreeNode("d", a)
    e = TreeNode("e", a)
    f = TreeNode("f", b)
    g = TreeNode("g", b)
    h = TreeNode("h", e)

    '''
        # Gives this tree
        # a
        # |   \  \   \
        # b    c  d   e
        # | \         |
        # f  g        h
    '''

    evc_node = a 

    '''
    a is root of EVC tree
    '''

    test_graph = evc_graph(evc_node)
    test_graph.image_visualize('./tmp/graph.png')
