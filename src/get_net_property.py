import os
import argparse
import networkx as nx
import json
import powerlaw
import matplotlib.pyplot as plt

def get_avg_mixing_parameter(G,communities):
    mixing_parameter = 0.0
    # mixing parameter = ratio of external degree to total degree
    for node in G.nodes():
        external_degree = 0.0
        nodecom = communities[node]
        nodedeg = G.degree(node)
        for neigh in G.neighbors(node):
            if nodecom!=communities[neigh]:
                external_degree += 1
        mixing_parameter += external_degree/nodedeg
    mixing_parameter = round(mixing_parameter/len(G.nodes()),2)    
    return mixing_parameter
    
def get_power_law_exponent(data,File):
    results = powerlaw.Fit(data,discrete = True)
    fig = plt.figure()
    ax = plt.subplot(111)
    fig=ax.hist(data)
    plt.savefig(File+".png")
    return round(results.power_law.alpha,2)

def intra_cluster_dens(connections, comsize):
    '''
    return a dictionary in which keys are clusters and values are intra cluster density (#internal edges of C / (n_c*(n_c-1)/2))
    '''
    intra_dens = {}
    for com in comsize:
        if com in connections and com in connections[com]:
            intra_dens[com] = connections[com][com] / (comsize[com] * (comsize[com]-1))
        else:
            intra_dens[com] = 0
    return intra_dens

def get_connections(G,communities):
    connections = {}
    comsize = {}
    for node in G:
        com = communities[node]
        if com not in comsize:
            comsize[com] = 1
        else:
            comsize[com] += 1    
        if com not in connections:
            connections[com] = {}
        for neigh in G.neighbors(node):
            neighcom = communities[neigh]
            if neighcom not in connections[com]:
                connections[com][neighcom] = 1.0
            else:
                connections[com][neighcom] += 1.0 
                
    return connections, comsize
        
def get_degree_dist(G):
    degrees = {}
    for node in G.nodes():
        deg = G.degree(node)
        if deg in degrees:
            degrees[deg] += 1
        else:
            degrees[deg] = 1
            
    return degrees            
def loadNet(path):
    
    communities={}
    G = nx.Graph()

    if(not os.path.isfile(path)):
        print("Error: file '" + path +"not found")
        exit(-1)
    if(not os.path.isfile(path + ".community")):
        print("Error: file '" + path + ".community' not found")
        exit(-1)    

    # create network
    with open(path) as f:   
        for line in f.readlines():
            v1 = int(line.split(" ")[0])
            v2 = int(line.split(" ")[1])
            G.add_node(v1)
            G.add_node(v2)
            G.add_edge(v1, v2)
            
    # Read the communities    
    with open(path + ".community") as f:
        for line in f.readlines():
            v = int(line.split(" ")[0])
            c = int(line.split(" ")[1])
            communities[v]=c

    return G,communities    
    
    		
def __main():
    parser = argparse.ArgumentParser(description = 'compute and store the properties of networks')
    parser.add_argument('nets', help='Path to networks')
    args = parser.parse_args()
    
    for File in os.listdir(args.nets):
        data={}
        if File.endswith(".community"):
                continue
        with open("properties/"+File, 'w') as output:
            G, communities = loadNet(args.nets+"/"+File)
            connections, comsize = get_connections(G,communities)   
            intra_dens = intra_cluster_dens(connections, comsize)
            data['nodes'] = str(len(G.nodes()))
            data['edges'] = str(len(G.edges()))
            data['communities'] = str(len(set(communities.values())))
            data['degrees'] = get_degree_dist(G)
            data['avg_clustering_coef'] = str(round(nx.average_clustering(G),2))
            data['degree_assortativity'] = str(round(nx.degree_assortativity_coefficient(G),2))
            if nx.is_connected(G):
                data['average_shortest_path'] = str(round(nx.average_shortest_path_length(G),2))
                data['diameter'] = str(nx.diameter(G))
            else:    
                data['average_shortest_path'] = '-'
                data['diameter'] = '-'
            
            data['intra_cluster_density'] = intra_dens
            data['degree_exp'] = str(get_power_law_exponent(list(G.degree().values()),File))
            data['comsize_exp'] = str(get_power_law_exponent(list(comsize.values()),File))
            data['avg_degree'] = str(round(2.0*len(G.edges())/len(G.nodes()),2))
            data['max_degree'] = str(max(G.degree().values()))
            data['min_comsize'] = str(min(comsize.values()))
            data['max_comsize'] = str(max(comsize.values()))
            data['mixing_parameter'] = str(get_avg_mixing_parameter(G,communities))
            json.dump(data, output)
            #print(File)
            #print("degree exp "+str(get_power_law_exponent(list(G.degree().values()),File)))
            #print("comsize exp "+str(get_power_law_exponent(list(comsize.values()),File)))
            #print("avg degree "+ str(round(2.0*len(G.edges())/len(G.nodes()),2)))
            #print("max degree "+ str(max(G.degree().values())))
            #print("minc "+str(min(comsize.values())))
            #print("maxc "+str(max(comsize.values())))
            #print("avg mixing parameter "+str(get_avg_mixing_parameter(G,communities)))
            #print("avg clustering coef "+str(round(nx.average_clustering(G),2)))
            #print("-----------------")
if __name__ == "__main__":
    __main()
