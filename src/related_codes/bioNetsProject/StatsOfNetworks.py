import networkx as nx
import graph_tool.all as gt
import matplotlib.pyplot as plt


def fitPowerlaw(data):
    '''
        tests if a probability distribution fits a power law and plots the result
        see https://pypi.python.org/pypi/powerlaw and http://tuvalu.santafe.edu/~aaronc/powerlaws/
    '''
    import powerlaw
   # data = array([1.7, 3.2 ...]) #data can be list or Numpy array
    results = powerlaw.Fit(data)
    print 'power law fittet exponent: ', results.power_law.alpha, ' xmin:' , results.power_law.xmin
    R, p = results.distribution_compare('power_law', 'lognormal')
    #Loglikelihood ratio of the two distributions' fit to the data. If greater than 0, the first distribution is preferred. 
    #If less than 0, the second distribution is preferred.
    powerlaw.plot_pdf(data, color='b')
    powerlaw.plot_pdf(data, linear_bins=True, color='r')
    plt.show()


def printGraphStats(net):
    '''
        computes basic characteristics of the given network
    '''
    print '>>>>> statistics for ', net['name'], ' ', net['type']
    G = net['graph']    
    
    if net['graph_format']== 'Networkx':   
        print 'Directed' if nx.is_directed(G) else 'UnDirected',
        print 'Weighted' if len(nx.get_edge_attributes(G,'weight') )>1 else 'UnWeighted',
        print ' Graph with ',   nx.number_of_nodes(G),'Nodes and ',  nx.number_of_edges(G), 'Edges '
        print 'Density of the Graph is ', nx.density(G), ' and it has ', 
        print  ' components', len(nx.connected_components(G.to_undirected())) 
        print 'Degree assortativity coefficient ', nx.degree_assortativity_coefficient(G)
        
        fitPowerlaw(nx.degree(G).values())
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
    elif net['graph_format'] == 'Graph-tool':
        #print G
        # for v in G.vertices():
        #    print v,
        print 'pseudo_diameter: ', gt.pseudo_diameter(G,source=G.vertex(1))[0] #Compute the pseudo-diameter of the graph.
        #deg = G.degree_property_map("total")
        #ebet = gt.betweenness(G)[1]
    else: 
        raise ValueError(net['graph_format'] + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
        
        
        
        