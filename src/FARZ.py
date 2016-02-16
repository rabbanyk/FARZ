import random
import bisect
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import draw_plots as mplt


def random_choice(values, weights=None, size = 1, replace = True):
    if weights is None:
        i = int(random.random() * len(values))
    else :
        total = 0
        cum_weights = []
        for w in weights:
            total += w
            cum_weights.append(total)
        x = random.random() * total
        i = bisect.bisect(cum_weights, x)
    
    if size <=1: 
        if len(values)>i: return [values[i]] 
        else: return None
    else: 
        cval = [values[j] for j in range(len(values)) if replace or i<>j]
        if weights is None: cwei=None 
        else: cwei = [weights[j] for j in range(len(weights)) if replace or i<>j]
        tmp= random_choice(cval, cwei, size-1, replace)
        tmp.append(values[i])
        return tmp 

class Comms:
     def __init__(self, k):
         self.k = k
         self.groups = [[] for i in range(k)]
         self.memberships = {}
         
     def add(self, cluster_id, i, s = 1):
         if i not in  [m[0] for m in self.groups[cluster_id]]:
            self.groups[cluster_id].append((i,s)) 
            if i in self.memberships:
                self.memberships[i].append((cluster_id,s))
            else:
                self.memberships[i] =[(cluster_id,s)] 
     def write_groups(self, path):
         with open(path, 'w') as f:
             for g in self.groups:
                 for i,s in g:
                    f.write(str(i) + ' ')
                 f.write('\n')

             
            
class Graph:
    def __init__(self,directed=False, weighted=False):
        self.n = 0
        self.counter = 0
        self.max_degree = 0
        self.directed = directed
        self.weighted = weighted
        self.edge_list = [] 
        self.edge_time = []
        self.deg = []
        self.neigh =  [[]]
        return 

    def add_node(self):
        self.deg.append(0)
        self.neigh.append([])
        self.n+=1
    
    def weight(self, u, v):
        for i,w in self.neigh[u]:
            if i == v: return w
        return 0
    
    def is_neigh(self, u, v):
        for i,_ in self.neigh[u]:
            if i == v: return True
        return False
    
    def add_edge(self, u, v, w=1):
        if u==v: return
        if not self.weighted : w =1
        self.edge_list.append((u,v,w) if u<v or self.directed else  (v,u,w))
        self.edge_time.append(self.counter)
        self.counter +=1
        self.neigh[u].append((v,w))
        self.deg[v]+=w
        if  self.deg[v]>self.max_degree: self.max_degree = self.deg[v]
        
        if not self.directed: #if directed deg is indegree, outdegree = len(negh)
            self.neigh[v].append((u,w))
            self.deg[u]+=w
            if  self.deg[u]>self.max_degree: self.max_degree = self.deg[u]

        return 
    
     
    def to_nx(self):
        G=nx.Graph()
        for u,v, w in self.edge_list:
            G.add_edge(u, v)
#         G.add_edges_from(self.edge_list)
        return G
    
    def to_nx(self, C):
        G=nx.Graph()
        for i in range(self.n):
            G.add_node(i, {'c':int(C.memberships[i][0][0])})
        for i in range(len(self.edge_list)):
#         for u,v, w in self.edge_list:
            u,v, w = self.edge_list[i]
            G.add_edge(u, v, weight=w, capacity=self.edge_time[i])
#         G.add_edges_from(self.edge_list)
        return G
    
    def to_ig(self):
        G=ig.Graph()
        G.add_edges(self.edge_list)
        return G 
 
 
    def write_edgelist(self, path):
         with open(path, 'w') as f:
             for i,j,w in self.edge_list:
                 f.write(str(i) + '\t'+str(j) + '\n')

 
def Q(G, C):
    q = 0.0
    m = 2 * len(G.edge_list)
    for c in C.groups:
        for i,_ in c:
            for j,_ in c:
                q+= G.weight(i,j) - (G.deg[i]*G.deg[j]/(2*m))
    q /= 2*m
    return q

def common_neighbour(i, G, normalize=True):
    p = {}
    for k,wik in G.neigh[i]:
        for j,wjk in G.neigh[k]:
            if j in p: p[j]+=(wik * wjk) 
            else: p[j]= (wik * wjk)
    if len(p)<=0 or not normalize: return p
    maxp = p[max(p, key = lambda i: p[i])]
    for j in p:  p[j] = p[j]*1.0 / maxp
    return p

def co_membership(i, C):
    p = {}
    for k,uik in C.memberships[i]:
        for j,ujk in C.groups[k]:
            if j in p: p[j]+=(uik * ujk)
            else: p[j] =  (uik * ujk)
    return p

def choose_community(i, G, C, alpha, beta, gamma, epsilon):
    mids =[k for  k,uik in C.memberships[i]]
    if random.random()< beta: #inside
        cids = mids
    else:     
        cids = [j for j in range(len(C.groups)) if j not in mids] #:  cids.append(j)

    return cids[ int(random.random()*len(cids))] if len(cids)>0 else None

# def dd(i,j, gamma, G):
#     if G.max_degree<=0: return 0 
#     delta = ((G.deg[j] -G.deg[i])*1.0/G.max_degree )**2
#     return (1-delta) if gamma > 0 else delta 
#     delta = (G.deg[j] -G.deg[i]) *(1 if G.deg[j]>G.deg[i] else -1)
#     return delta 
def degree_similarity(i, ids, G, gamma, normalize = True):
    p = [0]*len(ids)
    for ij,j in enumerate(ids):
        p[ij] =  (G.deg[j] -G.deg[i])**2
    if len(p)<=0 or not normalize: return p
    maxp = max(p)
    if maxp==0: return p
    p = [pi*1.0/maxp if gamma<0 else 1-pi*1.0/maxp for pi in p]
    return p

def combine (a,b,alpha,gamma, method = 'multi'):
    if method == 'mix':     return (a * alpha - b* gamma) if gamma<0 else a * alpha 
    if method == 'multi':  return (a**alpha) / ((b+1)**gamma)

    if gamma<0: gamma *=-1
    t=1
    if method == 'soft':    return (math.exp((a * alpha + b* gamma)/t))-1  
#     if method == 'geomean': return math.pow( a**alpha * b** gamma, 1.0/(alpha*gamma))  
    if method == 'linear':  return a * alpha + b* gamma
    if method == 'multi1':  return (a**alpha) * (b**gamma)
#     if method == 'hard':    return (math.exp(a * alpha) + math.exp(b* gamma))*.5 -1  #same problem as linear
#     elif method == 'multi2':  return ((a+1)**alpha) * ((b+1)**gamma) #bad
#     elif method == 'multi3':  return ( a * alpha + b* gamma )**2 #same problem as linear 

def choose_node(i,c, G, C, alpha, beta, gamma, epsilon):
    ids = [j for j,_ in C.groups[c] if j !=i ]
    #   also remove nodes that are already connected from the candidate list
    for k,_ in G.neigh[i]: 
        if k in ids: ids.remove(k) 

    norma = False
    cn = common_neighbour(i, G, normalize=norma)
    dd = degree_similarity(i, ids, G, gamma, normalize=norma)
    
    p = [epsilon for j in range(len(ids))]
    for ind in range(len(ids)):
        j = ids[ind]
        p[ind] = combine(cn[j] if j in cn else 0 , dd[ind], alpha, gamma) + epsilon
        
    if(sum(p)==0): return  None
    tmp = random_choice(range(len(p)), p , size=1, replace = False)
    # TODO add weights /direction/attributes
    if tmp is None: return  None
    return ids[tmp[0]], p[tmp[0]]

 
def connect_neighbor(i, j, pj, c, b,  G, C, beta):
    if b<=0: return 
    ids = [id for id,_ in C.groups[c]]
    for k,wjk in G.neigh[j]:
        if (random.random() <b and k!=i and (k in ids or random.random()>beta)):
            G.add_edge(i,k,wjk*pj)
                    
def connect(i, b,  G, C, alpha, beta, gamma, epsilon):
    #Choose community
    c = choose_community(i, G, C, alpha, beta, gamma, epsilon)
    if c is None: return
    #Choose node within community
    tmp = choose_node(i, c, G, C, alpha, beta, gamma, epsilon)
    if tmp is None: return
    j, pj = tmp 
    G.add_edge(i,j,pj)
    connect_neighbor(i, j, pj , c, b,  G, C, beta)
            
def select_node(G, method = 'uniform'):
    if method=='uniform':   
        return int(random.random() * G.n) # uniform
    else:
        if method == 'older_less_active': p = [(i+1) for i in range(G.n)] # older less active
        elif method == 'younger_less_active' :  p = [G.n-i for i in range(G.n)] # younger less active
        else:  p = [1 for i in range(G.n)] # uniform
        return  random_choice(range(len(p)), p , size=1, replace = False)[0]

def assign(i, C, e=1, o=1):
    p = [e +len(c) for c in C.groups]
    ids = random_choice(range(C.k),p, size=o)
    for id in ids: #todo add strength for fuzzy
        C.add(id, i)
    return 

def realize(n, m,  k, b=0.0,  alpha=0.4, beta=0.5, gamma=0.1, phi=1, o=1, epsilon = 0.0000001):
    print n, m , k, b, alpha, beta, gamma, phi, o, epsilon
    G =  Graph()
    C = Comms(k)
        
    for i in range(n):
#         if i%10==0: print '-- ',G.n, len(G.edge_list)
        G.add_node()
        assign(i, C, phi, o)
        connect(i,b, G, C, alpha, beta, gamma, epsilon)
        for e in range(m):
            j = select_node(G) 
            connect(j, b, G, C, alpha, beta, gamma, epsilon)        
    return G,C


def props():
    import plotNets as pltn
    import matplotlib as mpl
    mpl.rcParams['axes.unicode_minus']=False
    
    graphs = []
    names = []
    
    params = {"n":1000, "k":4, "m":4,  "b":0.0, "alpha":1, "beta":.8, "phi":1, "o":1, "epsilon":0.0000001}
    for alp, gam in [(0.5,0.5), (0.8,0.2), (.5,-0.5), (0.2,-0.8)]:
        params[ "alpha"]=alp
        params[ "gamma"]=gam
        print str(params)
        G, C =realize(**params)
        print 'n=',G.n,' e=', len(G.edge_list)
        print 'Q=',Q(G,C)
        G = G.to_nx(C)
        pltn.printGraphStats(G)
        graphs.append(G.to_undirected())
        # name = 'F'+str(params)
        name = '$\\alpha ='+str(params[ "alpha"]) +',\; \\gamma='+str(params[ "gamma"])+"$"
        names.append(name)
        nx.write_gml(graphs[-1], "farz-"+str(params[ "alpha"])+str(params[ "gamma"])+'.gml')
    
    pltn.plot_dists(graphs,names)

def generate( vari ='beta', arange =(0.5,1,0.05), repeat = 2, path ='.', format ='gml', directed =False,
               params= {"n":1000, "k":4, "m":4,  "b":0.0, "alpha":0.5, "beta":.8, "gamma":-0.5,"phi":1, "o":1, "epsilon":0.0000001}):
    import os 
    if not os.path.exists(path+'/'): os.makedirs(path+'/')
    for i,var in enumerate(np.arange(arange[0],arange[1]+arange[2],arange[2])): #to be inclusive in endpoints
        for r in range(repeat):
            params[vari] = var
            print 's',i+1, r+1, str(params)
            G, C =realize(**params)
            print 'n=',G.n,' e=', len(G.edge_list)
            name = 'S'+str(i+1)+'-network'+str(r+1)
            if format == 'gml':
                G = G.to_nx(C)
                if not directed: G = G.to_undirected()
                nx.write_gml(G, path+'/'+name+'.gml')
            if format == 'list': 
                G.write_edgelist(path+'/'+name+'.dat')
                C.write_groups( path+'/'+name+'.lgt')
                
                
paramset = {'55':{"n":1000, "k":4, "m":4,  "b":0.0, "alpha":0.5,"gamma":0.5, "beta":.8, "phi":1, "o":1, "epsilon":0.0000001},
            '82':{"n":1000, "k":4, "m":4,  "b":0.0, "alpha":0.8,"gamma":0.2, "beta":.8, "phi":1, "o":1, "epsilon":0.0000001},
            '5-5':{"n":1000, "k":4, "m":4,  "b":0.0, "alpha":0.5,"gamma":-0.5, "beta":.8, "phi":1, "o":1, "epsilon":0.0000001},
            '2-8':{"n":1000, "k":4, "m":4,  "b":0.0, "alpha":0.2,"gamma":-0.8, "beta":.8, "phi":1, "o":1, "epsilon":0.0000001}
            }

# for sett in paramset:
#     generate('beta', repeat =10, path='./vbeta'+sett+'/data',params=paramset[sett])

#vari k 2 to 100 :: too few, too many communities
# for sett in paramset:
#     generate('k', arange=(2,50,5), repeat =10, path='./vk'+sett+'/data',params=paramset[sett])

#vari phi, 1 to 1000 :: sizes of communities more balanced

# for sett in paramset:
#     generate('phi', arange=(1,100,10), repeat =10, path='./vp2'+sett+'/data',params=paramset[sett])

#vari m 2 to 10 :: too sparce too dense
# for sett in paramset:
#     generate('m', arange=(2,11,1), repeat =10, path='./vm'+sett+'/data',params=paramset[sett])

#vari o for overlapping methods 1 to 5 ::: CM methods should change

for sett in paramset:
    params= paramset[sett]
    params['k'] = 20
    params['m'] = 6
    params['beta'] = 0.9
    generate('o', arange=(1,10,1), repeat =1, path='./vo'+sett+'/data', format = 'list',params=params)





