import igraph as ig
import networkx as nx
import numpy as np


class Graph:
    def __init__(self, n, edge_list, directed=False):
        self.edge_list = edge_list 
        self.n = n
        self.directed=directed
        #         self.C = None
        #         self.Cid = None
        self.rewirings_count = 0
        
        self.deg = [0]* n
        self.neigh =  [[] for i in range(n)]
        for e in edge_list:
            u , v = e
            self.neigh[u].append(v)
            self.deg[v]+=1 
            if not self.directed: #if directed deg is indegree, outdegree = len(negh)
                self.neigh[v].append(u)
                self.deg[u]+=1            
        return 

    def add_edge(self, u, v):
        self.edge_list.append((u,v) if u<v else (v,u))
        self.neigh[u].append(v)
        self.deg[v]+=1
        if not self.directed: 
            self.neigh[v].append(u)
            self.deg[u]+=1
        self.rewirings_count +=1
        return 
    
    def remove_edge(self,u, v):
        self.edge_list.remove((u,v) if u<v else (v,u))
        self.neigh[u].remove(v)
        if not self.directed: 
             self.neigh[v].remove(u)
             self.deg[u]-=1
        self.deg[v]-=1   
        self.rewirings_count +=1
        return 

    def to_nx(self):
        G=nx.Graph()
        G.add_edges_from(self.edge_list)
        return G
    
    def to_ig(self):
        G=ig.Graph()
        G.add_edges(self.edge_list)
        return G 


def albert_barabasi(params):
    g = ig.Graph.Barabasi(**params)
    return Graph(g.vcount(), g.get_edgelist())

def forest_fire(params):    
    g = ig.Graph.Forest_Fire(**params)
    return Graph(g.vcount(), g.get_edgelist())

def erdos_renyi(params):    
    g = ig.Graph.Erdos_Renyi(**params)
    return Graph(g.vcount(), g.get_edgelist())


def configuration_model_from_sequence(S, multilinks = False, selfloop = False):
    ''' 
        would generate a graph from the given sequence, with the configuration model, i.e. forms edges with uniform probability
        the resulted nodes will have degrees <= given sequence -> hence works with any input, even if not a valid degree sequence
        # todo: add direction and weights
    '''
    edge_list = []
    max_itr = len(S)
    itr = 0
    Q = [i for i in range(0, len(S)) if S[i]!=0 ]
    while len(Q)>(1 if not selfloop else 0):
        itr+=1
        i, j = np.random.choice(Q, size =2, replace = False if not selfloop else True)
        e = (i,j) if i<j else (j,i)
        # woudn't terminate if all pairs in Q are connected in edge_list
        if not e in edge_list or len(Q)*(len(Q)-1) < 2* len(edge_list) or itr>max_itr: 
            edge_list.append(e)
            S[i]-=1
            S[j]-=1
            itr = 0
            if S[i]==0: Q.remove(i)
            if S[j]==0: Q.remove(j)
    
    return edge_list if multilinks else list(set(edge_list))

