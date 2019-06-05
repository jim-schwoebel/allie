'''
Draw graphs to visualize audio data with network theory.

Network theory - https://github.com/networkx/networkx
Documentation - https://networkx.github.io/documentation/networkx-1.10/tutorial/tutorial.html#drawing-graphs

>>> import networkx as nx
>>> G = nx.Graph()
>>> G.add_edge('A', 'B', weight=4)
>>> G.add_edge('B', 'D', weight=2)
>>> G.add_edge('A', 'C', weight=3)
>>> G.add_edge('C', 'D', weight=4)
>>> nx.shortest_path(G, 'A', 'D', weight='weight')
['A', 'B', 'D']
'''
import networkx as nx
import numpy.linalg
import matplotlib.pyplot as plt

n = 1000 # 1000 nodes
m = 5000 # 5000 edges
G = nx.gnm_random_graph(n,m)

L = nx.normalized_laplacian_matrix(G)
e = numpy.linalg.eigvals(L.A)
print("Largest eigenvalue:", max(e))
print("Smallest eigenvalue:", min(e))
plt.hist(e,bins=100) # histogram with 100 bins
plt.xlim(0,2)  # eigenvalues between 0 and 2
plt.show()

# mlab.show() # interactive window


#TUTORIAL
# can do this for many audio features (mfcc coefficients)
##
##G = nx.Graph(day="Friday")
##G.add_nodes_from(range(100,110))
##G.add_node("spam")
##G.remove_nodes_from("spam")
##
##G.add_path(range(100,110))
##
##G.add_edge('A', 'B', weight=1)
##G.add_edge('A', 'C', weight=1)
##G.add_edge('C', 'D', weight=1)
##G.add_edge('B', 'D', weight=1)
##
##G.remove_edge('B','D')
##
##nx.draw(G)
##plt.savefig("path.png")
##
##
##print(G.edges(data='weight'))
##print(G.number_of_nodes())
##print(G.number_of_edges())

