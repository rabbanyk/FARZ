import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import *
import graph_tool.all as gt
import sys


def printGraphStats(G):
    '''
        computes basic characteristics of the given network
    '''
    print 'Directed' if nx.is_directed(G) else 'UnDirected',
    print 'Weighted' if len(nx.get_edge_attributes(G,'weight') )>1 else 'UnWeighted',
    print ' Graph with ',   nx.number_of_nodes(G),'Nodes and ',  nx.number_of_edges(G), 'Edges '
    print 'Density of the Graph is ', nx.density(G), ' and it has ', 
    print  ' components', len(nx.connected_components(G.to_undirected())) 
    print 'Degree assortativity coefficient ', nx.degree_assortativity_coefficient(G)
    
#        fitPowerlaw(nx.degree(G).values())
    # to avoid: networkx.exception.NetworkXError: Graph not connected: infinite path length
    if len(nx.connected_components(G.to_undirected())) ==1:
        G = G.to_undirected()
        print "Additional Stats available for connected graphs: "
        print 'Average shortest path length : ', nx.average_shortest_path_length(G.to_undirected())
        print("diameter: %d" % nx.diameter(G))
        print("periphery: %s" % nx.periphery(G))
        print("center: %s" % nx.center(G))
        print("eccentricity: %s" % nx.eccentricity(G))
        print("radius: %d" % nx.radius(G))

def plotDegreeSeq(G):
    kwargs = {'alpha':0.4, 'edge_color':'b', 'font_size':8}#, 'node_size':deg, 'node_color':deg}
    degree_hist=nx.degree_histogram(G)
    
    print 'Degrees hist: ', len(degree_hist),degree_hist
    plt.plot(degree_hist,'g-',marker='.')

    plt.title("Degree distribution plot")
    plt.ylabel("frequency")
    plt.xlabel("degree")
    #plt.savefig("degree_histogram.png")
    plt.show()

def plotGraph(G):
    deg = nx.degree(G).values()
    kwargs = {'alpha':0.4, 'edge_color':'b', 'font_size':8, 'node_size':deg, 'node_color':deg}
   
    fig = plt.figure()
    ''' fig.add_subplot(231).set_title("circular layout")
    nx.draw_circular(G, **kwargs)    #Draw the graph G with a circular layout. 
    fig.add_subplot(232).set_title("random layout")
    nx.draw_random(G, **kwargs)    #Draw the graph G with a random layout.
    fig.add_subplot(233).set_title("spectral layout")
    nx.draw_spectral(G, **kwargs)   #Draw the graph G with a spectral layout.
    fig.add_subplot(234).set_title("spring layout")'''
    nx.draw_spring(G, **kwargs)    #Draw the graph G with a spring layout.
    '''fig.add_subplot(235).set_title("shell layout")
    nx.draw_shell(G, **kwargs)    #Draw networkx graph with shell layout.
    fig.add_subplot(236).set_title("graphviz layout")
    nx.draw_graphviz(G, prog='neato', **kwargs)    #Draw networkx with graphviz.
    '''
    plt.show()



def findCommunities(path):
    '''
        finds communities in the given network
    '''
    #  b = gt.community_structure(G, 10000, 10)
    G = gt.load_graph(path)
    print G.num_vertices()
    spins = gt.community_structure(G, 10000, 20, t_range=(5, 0.1),  history_file="community-history1")
    print  spins
    #gt.graph_draw(G, vertex_fill_color=spins, output_size=(420, 420))
    
    print gt.modularity(G, spins)
     
     
def findCommunities2(g):     
    state = gt.BlockState(g, B=5, deg_corr=True)
    print 1
    for i in range(1000):        # remove part of the transient
        ds, nmoves = gt.mcmc_sweep(state)
    print 2
    for i in range(1000):
        ds, nmoves = gt.mcmc_sweep(state, beta=float("inf"))    
    print 3
    b = state.get_blocks()
    print 4
    gt.graph_draw(g, vertex_fill_color=b, vertex_shape=b, output="test2.pdf")
    print 5
    bg, bb, vcount, ecount, avp, aep = gt.condensation_graph(g, b, self_loops=True)
    print 6
    print 
    pv = 2
    gt.graph_draw(bg, vertex_fill_color=bb, vertex_shape="pie", vertex_pie_fractions=pv,
                  vertex_size=gt.prop_to_size(vcount, mi=40, ma=100),
                  edge_pen_width=gt.prop_to_size(ecount, mi=2, ma=10),
                  output="test22.pdf")
    
def findCommunities3(g):     
    g = gt.collection.data["polbooks"]
    state = gt.BlockState(g, B=3, deg_corr=True)
    pv = None
    for i in range(1000):        # remove part of the transient
     ds, nmoves = gt.mcmc_sweep(state)
    for i in range(1000):
        ds, nmoves = gt.mcmc_sweep(state)
        pv = gt.collect_vertex_marginals(state, pv)
    print pv
    for p in pv:
        print p    
    gt.graph_draw(g, pos=g.vp["pos"], vertex_shape="pie", vertex_pie_fractions=pv, output="polbooks_blocks_soft.pdf")
 
        
g = gt.load_graph('Rice31.xml')
#g = gt.collection.data["polbooks"]
findCommunities3(g)
    
