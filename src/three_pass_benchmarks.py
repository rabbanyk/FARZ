import random
import bisect
import numpy as np
from network_models import *

def generalize_three_pass(network_model,  assign_nodes, overlay_communities, g_params, c_params):
    G = network_model(g_params)
#     print_seq_stats( '\t\t network_generated', G.deg)
    return generalize_three_pass_network(G, assign_nodes, overlay_communities,  c_params )

def generalize_three_pass_network(G,  assign_nodes, overlay_communities, c_params):
    C = assign_nodes(G,  c_params)
    print_seq_stats('\t\t node_assigned', [len(c) for c in C[0]])
    return overlay_communities(G, C,  c_params)

def print_seq_stats(msg, S):
    print msg, ':::\t len: ',len(S),' min: ', np.min(S) ,' avg: ',np.mean(S),' max: ',  np.max(S),' sum: ',  np.sum(S)



def assign_CN(G, c_params):
    def cn_prob(G, v, C, ec,  Cid):
        p = [0.1]*(len(C))
        for u in G.neigh[v]:
            if Cid[u]>=0: p[Cid[u]]+=1
        p= [p[i] for i in ec] #remove communities that are full
        p= [i/sum(p) for i in p]#np.divide(p, np.sum(p))
#         print p
        return p
    return assign_LFR(G,  c_params, cn_prob)

def random_choice(values, weights=None, size = 1, replace = True):
    if weights is None:
        i = int(random.random() * len(values))
#         i = random.randrange(0,len(values))
#         res = random.choice(values)
    else :
        total = 0
        cum_weights = []
        for w in weights:
            total += w
            cum_weights.append(total)
        x = random.random() * total
        i = bisect.bisect(cum_weights, x)
#         print weights
        #res =  values[i]
    
    if size <=1: return values[i]
    else: 
#         print i,  values
        cval = [values[j] for j in range(len(values)) if replace or i<>j]
        if weights is None: cwei=None 
        else: cwei = [weights[j] for j in range(len(weights)) if replace or i<>j]
        
#         if not replace :  del values[i]
        return values[i], random_choice(cval, cwei, size-1, replace)
    
def assign_first_pass_original(G,mu, s, c, cid, prob):
#   assign nodes to communities
    for v in range(G.n):
#         pick a community at random
        ec = [i for i in range( len(c)) if s[i]>len(c[i])]

#         if prob is None:
#             p = None 
#         else: 
#             print prob
#             print ec
#             p = prob(G,v,c, cid)
#             p = [p[i] for i in ec]
#             p =  np.divide(p, np.sum(p))
        i = random_choice(ec, None if prob is None else prob(G,v,c,ec, cid))
#       assign to community if fits
        if s[i] >= (1- mu)*G.deg[v] :  
            c[i].append(v)
            cid[v] = i
    
def assign_first_pass_NE(G,mu, s, c, cid, prob):
    for v in range(G.n):
        if cid[v]==-1:
    #         pick a community at random
            ec = [i for i in range( len(c)) if s[i]>len(c[i])]
            i = random_choice(ec, None if prob is None else prob(G,v,c,ec, cid))

#             i = np.random_choice([i for i in range( len(c)) if s[i]>len(c[i])], 
#                           p = None if prob is None else prob(G,v,c, cid) )
            to_add = [v]
            marked = [0]*G.n
            marked[v] =1
            while len(to_add)>0 and len(c[i])< s[i]:
                v = to_add.pop(0)
        #       assign to community if fits
                if s[i] >= (1- mu)*G.deg[v] :  
                    c[i].append(v)
                    cid[v] = i
                    for u in G.neigh[v]: 
                        if marked[u]==0 and cid[u]==-1: 
                            to_add.append(u)
                            marked[u]=1

    
def assign_LFR(G, c_params, prob=None, first_pass= assign_first_pass_original,  max_itr = 1000):
    c_params['s_sum'] = G.n
    mu = c_params['mu']
#   determine capacity of communities
    d_max = np.max(G.deg)
#     print G.n, c_params['s_max'], d_max, (1-mu) *d_max 
    
    if  c_params['s_max']< (1-mu) *d_max :  c_params['s_max'] =(1-mu) * d_max
#     print c_params['s_max'], d_max
  
    s = sample_power_law(**c_params)
    c_max = max(s)
#     print_seq_stats('community_sizes_sampeled',s)

#   initialize the communities and community ids
    c = [[] for i in range(len(s))]
    cid = [-1] * G.n
    first_pass(G,mu, s, c, cid, prob)
    
#     print_seq_stats('1... ',[len(l) for l in c])
#   initialize the homeless queue
    H = [v for v in range(G.n) if cid[v]==-1]
   
    itr = 0
#   assign homeless nodes to communities
    while len(H)>0 and max_itr>itr:
        itr+=1
#         print itr
#       pick a community at random
        v = random_choice(H)
        ec = [i for i in range( len(c))]
        i = random_choice(ec,   None if prob is None else prob(G, v,c,ec, cid) )
        if s[i] >= min((1- mu)*G.deg[v], c_max) :
            c[i].append(v)
            cid[v]=i
            H.remove(v)
#             itr=0
#           kick out a random node
            if len(c[i])> s[i]:
                u = random_choice(c[i])
                c[i].remove(u)
                cid[u] = -1
                H.append(u)
                
    if len(H)>0: print "Failed in 2nd run"
    for v in H:
#       pick a community at random
        ec =[i for i in range( len(c))]
        i = random_choice(ec, None if prob is None else prob(G, v,c,ec, cid) )
        c[i].append(v)
        cid[v]=i
        
    return c, cid



def assign_NE(G, c_params, prob=None):
    return assign_LFR(G, c_params, prob=None, first_pass= assign_first_pass_NE)

def overlay_LFR(G, C, c_params):
    mu = c_params['mu']
    n= G.n
    deg = G.deg #degree_seq(n, edge_list)
    
    C, Cid = C
    # determine degree of each node and its between/outlink degree, 
    # i.e. number of edges that go outside its community
    db = [0]* n
#     d = [0] * n
#     neigh = [[] for i in range(0, n)]
    for e in G.edge_list:
        u , v = e
        if Cid[u] != Cid[v]:
            db[u]+=1
            db[v]+=1 
    # determine desired between changes
    for v in range(n):
        db[v] = np.floor(mu*G.deg[v] - db[v])
    dw = np.multiply(db, -1) 
    # rewire edges within communities
    for c in C:
        I = [v for v in c if dw[v]>0]
        # add internal edges
        while len(I)>=2:
            u, v = random_choice(I, size =2, replace = False)
            G.add_edge(u,v)
            dw[u]-=1
            dw[v]-=1
            if dw[u] ==0: I.remove(u)
            if dw[v] ==0: I.remove(v)
        # remove excess edges
        for v in c:
            if dw[v]<0:
                I = [u for u in G.neigh[v] if u in c and dw[u]<0]
                while len(I)>=1 and dw[v]<0:
                    u = random_choice(I)
                    G.remove_edge(u,v)
                    dw[u]+=1
                    dw[v]+=1
                    I.remove(u)
          
    # rewire edges between communities 
    for c in C:
        I = [v for v in c if db[v]>0]
        O = [v for v in range(n) if v not in c and db[v]>0]
        # add internal edges
        while len(I)>=1 and len(O)>=1:
            v = random_choice(I)
            u = random_choice(O)
            G.add_edge(u,v)
            db[u]-=1
            db[v]-=1
            if db[v] ==0: I.remove(v)
            if db[u] ==0: O.remove(u)
        # remove excess edges
        for v in c:
            if db[v]<0:
                O = [u for u in G.neigh[v] if u not in c and db[u]<0]
                while len(O)>=1 and db[v]<0:
                    u = random_choice(O)
                    G.remove_edge(u,v)
                    db[u]+=1
                    db[v]+=1
                    O.remove(u)
                  
    return G, C

def configuration_model(params):
    S = sample_power_law(**params)
#     print_seq_stats('degree_sampled',S)
    return Graph(len(S),configuration_model_from_sequence(S))


def sample_power_law( s_exp=2, n=None, s_avg=None, s_max=None, s_min=1, s_sum=None, discrete = True, **kwargs):
    S = None
    if  n is not None: # number of samples is fixed
        if s_avg is None and s_sum is not None: s_avg = s_sum*1.0/n
        # 1.0/np.random.power(exp+1, size=n) 
        S = np.array([]) 
        c = None
        while (len(S)<n):
            tmp =  np.random.pareto(s_exp-1, size = n - len(S))  
            tmp, c = scale_truncate(tmp , s_max, s_min, s_avg , c)
            S= np.hstack( (S,tmp))
        S = np.around(S).astype(int) if discrete  else S
    elif s_sum is not None: # cumulative s_sum of samples is fixed
        S = [] 
        while (np.sum(S)<s_sum):    
            tmp =  np.random.pareto(s_exp-1, size = int( s_sum - np.sum(S)))  
            tmp, c = scale_truncate(tmp , s_max, s_min ,s_avg, c=1)
            tmp = np.around(tmp).astype(int) if discrete else tmp
            for t in tmp:
                if t+np.sum(S) <= s_sum:   
                    S.append(t) 
                elif np.sum(S) < s_sum:
                    tmp =  s_sum - np.sum(S)
                    if tmp>=s_min:
                        S.append(tmp)
                    else:
                        shift = np.ceil(tmp*1.0/len(S))
                        for i in range(0,len(S)): 
                            if tmp>0:
                                S[i]+=shift
                                tmp-=shift
                else: break            
    
#         print S, np.sum(S), s_sum
    return S


    
def scale_truncate(S, max=None, min=None, avg=None, c= None):
#     print c
    if c==None:
        c = 1
        if avg is not None: 
            itr =0
            max_itr = 100
            while (itr<max_itr ):
                itr+=1
                c_2 = avg/np.mean(S) # - min if min is not None else 0
                S = np.multiply(S,c_2)
                c *= c_2
                if min is not None: S = S[S>min]
                if max is not None: S = S[S<max] 
    else:
        S = np.multiply(S,c)
        if min is not None: S = S[S>min]
        if max is not None: S = S[S<max] 
    return S, c



    

def LFR(n, k_avg, k_max, mu, c_min, c_max, k_exp=2, c_exp=1, assign= assign_LFR, model = configuration_model):
    if c_max < (1-mu) * k_max: 
        c_max = k_max
        print 'maximum size for communities adjusted to fit the node with largest degree'
    return generalize_three_pass(model, assign, overlay_LFR,
                                 g_params = {"n":n, "s_avg":k_avg, "s_max":k_max, "s_exp":k_exp}, 
                                 c_params= { "mu": mu, "s_min": c_min, "s_max":c_max, "s_exp":c_exp}  )
    
    
    
    