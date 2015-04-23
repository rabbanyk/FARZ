import networkx as nx
import graph_tool.all as gt
import matplotlib.pyplot as plt

#gt.graphviz_draw(G2, vcolor=deg, vorder=deg, elen=10, ecolor=ebet, eorder=ebet)  

def compareDifferentGraphLayoutsInNetworkX(G):
    deg = nx.degree(G).values()
    kwargs = {'alpha':0.4, 'edge_color':'b', 'font_size':8, 'node_size':deg, 'node_color':deg}
   
    fig = plt.figure()
    fig.add_subplot(231).set_title("circular layout")
    nx.draw_circular(G, **kwargs)    #Draw the graph G with a circular layout. 
    fig.add_subplot(232).set_title("random layout")
    nx.draw_random(G, **kwargs)    #Draw the graph G with a random layout.
    fig.add_subplot(233).set_title("spectral layout")
    nx.draw_spectral(G, **kwargs)   #Draw the graph G with a spectral layout.
    fig.add_subplot(234).set_title("spring layout")
    nx.draw_spring(G, **kwargs)    #Draw the graph G with a spring layout.
    fig.add_subplot(235).set_title("shell layout")
    nx.draw_shell(G, **kwargs)    #Draw networkx graph with shell layout.
    fig.add_subplot(236).set_title("graphviz layout")
    nx.draw_graphviz(G, prog='neato', **kwargs)    #Draw networkx with graphviz.
  
    plt.show()

def compareDifferentGraphLayoutsInGraphtool(G):
    
    #, output="graph-draw-sfdp.pdf")
    deg = G.degree_property_map("in")
    ebet = gt.betweenness(G)[1]
    
    kwargs = { 'vcolor':deg, 'vorder':deg, 'elen':10, 'ecolor':ebet, 'eorder':ebet}

  
    fig = plt.figure()

    fig.add_subplot(231).set_title("SFDP spring layout")
    gt.graph_draw(G, pos = gt.sfdp_layout(G), **kwargs)    #Obtain the SFDP spring-block layout of the graph.
    
    fig.add_subplot(232).set_title("Fruchterman-Reingold spring-block layout")
    gt.graph_draw(G, pos = gt.fruchterman_reingold_layout(G, n_iter=1000), **kwargs)    #Calculate the Fruchterman-Reingold spring-block layout of the graph.
    
    fig.add_subplot(233).set_title("ARF spring-block layout")
    gt.graph_draw(G,pos = gt.arf_layout(G, max_iter=0), **kwargs)    #Calculate the ARF spring-block layout of the graph.
    
    fig.add_subplot(234).set_title("radial layout")
    gt.graph_draw(G,pos=gt.radial_tree_layout(G, G.vertex(0)), **kwargs)   #Computes a radial layout of the graph according to the minimum spanning tree centered at the root vertex.
   
    fig.add_subplot(235).set_title("random layout")
    gt.graph_draw(G,pos = gt.random_layout(G, dim=3), **kwargs)    #Performs a random layout of the graph.
   
    fig.add_subplot(236).set_title("graphviz layout")
    gt.graphviz_draw(G, vcolor=deg, vorder=deg, elen=10, ecolor=ebet, eorder=ebet)  

    plt.show()

def plotDegreeSeq(G,G2):
    kwargs = {'alpha':0.4, 'edge_color':'b', 'font_size':8}#, 'node_size':deg, 'node_color':deg}

   # degree_sequence = sorted(list(degree(G).vaAnalyzeNetworks.findCommunitieslues()), reverse=True)
   # degree_sequence=sorted(degree(G).values(),reverse=True) # degree sequence
    #print 'Degrees seq: ',degree_sequence, len(degree_sequence)
    degree_hist=nx.degree_histogram(G)
    degree_hist2,_tmp=gt.vertex_hist(G2, 'total', bins=[0, 1], float_count=False)
    
    print 'Degrees hist: ', len(degree_hist),degree_hist
    print 'Degrees hist: ', len(degree_hist2),degree_hist2

  #  plt.loglog(degree_sequence,'b-',marker='o')
    plt.plot(degree_hist,'g-',marker='.')
    plt.plot(degree_hist2,'b-',marker='.')

    plt.title("Degree distribution plot")
    plt.ylabel("frequency")
    plt.xlabel("degree")
    
    # draw graph in inset
    plt.axes([0.45,0.45,0.45,0.45])
    #Gcc= G#connected_component_subgraphs(G)[0]
    #pos=spring_layout(Gcc)
    plt.axis('off')
    #nx.draw_networkx_nodes(Gcc,pos,node_size=20)
    #nx.draw_networkx_edges(Gcc,pos,alpha=0.4)
    nx.draw_spring(G, **kwargs)
    #plt.savefig("degree_histogram.png")
    plt.show()



#https://www.udacity.com/wiki/creating%20network%20graphs%20with%20python

#compareDifferentGraphLayoutsInGraphtool(G2)
#compareDifferentGraphLayoutsInNetworkX(G)
#plotDegreeSeq(G,G2)