import networkx as nx
#import community
import numpy as np
import scipy as sp
from numpy.random import *
import graph_tool.all as gt
import matplotlib.pyplot as plt
import sys


def findCommunities(net):
    '''
        finds communities in the given network
    '''
    print '>>>>> communities for ', net['name'], ' ', net['type']
    G = net['graph']    
    
    if net['graph_format']== 'Networkx':   
        print 'we will use the graph tool here'
        # not available in Networkx
        '''  partition = community.best_partition(G)
        print 'modularity: ' , community.modularity(partition, G)
        
        size = float(len(set(partition.values())))
        pos = nx.spring_layout(G)
        count = 0.
        for com in set(partition.values()) :
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20, node_color = str(count / size))
        nx.draw_networkx_edges(G,pos, alpha=0.5)
        plt.show()
        '''
        #if len(nx.connected_components(G.to_undirected())) ==1:
         #   G = G.to_undirected()
         
    elif net['graph_format'] == 'Graph-tool':
      #  b = gt.community_structure(G, 10000, 10)
        spins = gt.community_structure(G, 10000, 20, t_range=(5, 0.1),  history_file="community-history1")
        print  spins
        gt.graph_draw(G, vertex_fill_color=spins, output_size=(420, 420))
        
        print gt.modularity(G, spins)
    else: 
        raise ValueError(net['graph_format'] + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
        


def computeCentralities(net):
    '''
        computes centralities of nodes in the given network
    '''
    print '>>>>> centralities for ', net['name'], ' ', net['type']
    G = net['graph']    
    
    if net['graph_format']== 'Networkx':   
        print 'degree centrality with Networkx'
        H = G
        kwargs = {'alpha':0.8, 'edge_color':'k', 'font_size':8}

       # try:
        plt.figure(figsize=(8,8))
        # with nodes colored by degree sized by population
        deg=[float(H.degree(v))*10 for v in H]
        nx.draw(H,
             node_size=deg,
             node_color=deg,
             with_labels=True, **kwargs)
        plt.show()
            # scale the axes equally
       #     plt.xlim(-5000,500)
        #    plt.ylim(-2000,3500)
    
        #except:
         #   pass
     
    elif net['graph_format'] == 'Graph-tool':
        g = gt.GraphView(G, vfilt=gt.label_largest_component(G))
        pos = None
        pos = gt.graph_draw(g, vertex_fill_color=bm, edge_color=[0, 0, 0, 0.3],
                            output_size =(250,250), output="img/blockmodel.png")
        #
        for centrality in ['degree','betweenness', 'closeness','pagerank', 'eigenvector', 'katz']:
            # G.list_properties()
            # try:
           
            print centrality, ' centrality with Graph-tool'
           
            if centrality =='degree':
                c = g.degree_property_map("total")
            else:
                c = getattr(gt, centrality)(g) #betweenness(G) #.pagerank(G) .closeness(G) .eigenvector(G) ..katz(G) 
                if centrality == 'betweenness': c = c[0]
                if centrality == 'eigenvector': c= c[1]
          #  print c.get_array()
            
            #print np.min(c.get_array()), np.max(c.get_array())
            nc = gt.prop_to_size(c, mi=2, ma=4)
         #   print nc.get_array()
            
            pos = gt.graph_draw(g, 
                                pos = pos, output = 'img/'+centrality+'.png',
                                output_size =(250,250), edge_color=[0, 0, 0, 0.3],
                                 vertex_fill_color=nc, #  vertex_text=g.vertex_index, 
                     vertex_size= nc 
                     )#, vorder=c, fit_view=True)[0] , output = filename
            #except:
             #   print "Unexpected error:", sys.exc_info()
              #  pass
       
        '''try:  
            print 'pagerank centrality with Graph-tool'

            pr = gt
            gt.graph_draw(G, pos=G.vp["pos"], vertex_fill_color=pr,  vertex_text=G.vertex_index, 
                          vertex_size=gt.prop_to_size(pr, mi=5, ma=15),  vorder=pr)
        except:
            pass
        '''
      
        
    else: 
        raise ValueError(net['graph_format'] + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
        
def corr(a, b):
    if a == b:
        return 0.999
    else:
        return 0.001


gra, bm = gt.random_graph(500, lambda: poisson(10), directed=False,
                         model="blockmodel-traditional",
                         block_membership= lambda: randint(10),
                         vertex_corr=corr)

#, output="blockmodel.pdf")
'''

def sample_k(max):
    accept = False
    while not accept:
        k = np.random.randint(1,max+1)
        accept = np.random.random() < 1.0/k
    return k
g = gt.random_graph(200, lambda: sample_k(40), model="probabilistic",
                    vertex_corr=lambda i, k: 1.0 / (1 + abs(i - k)), directed=False,
                   n_iter=100)

g = gt.collection.data["karate"]
'''
computeCentralities({'graph':gra, 'name':'randomBl', 'type':'social', 'graph_format':'Graph-tool'})
    