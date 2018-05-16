import os
import argparse
import networkx as nx
import json
import powerlaw
import matplotlib.pyplot as plt
import FARZ
import numpy as np
import cPickle as pickle

def binary_search(FARZparameters, targetparam, targetvalue):
    lower = 0
    upper = 1.0
    min_dif = 1
    best_val = 0.0
    while upper-lower >= 0.05:
        mid = (lower + upper) / 2
        #print("mid "+str(mid))
        FARZparameters.append('--'+targetparam)
        FARZparameters.append(str(mid))
        G,C = FARZ.FARZ(FARZparameters)
        if targetparam == 'alpha':
            val = round(nx.average_clustering(G),2)
        elif targetparam == 'gamma':
            val = round(nx.degree_assortativity_coefficient(G),2)
        #print("target "+str(targetvalue))        
        #print("val "+str(val))   
        if abs(targetvalue - val) < min_dif:
            min_dif = abs(targetvalue - val)
            best_val = mid
        if abs(targetvalue - val) < 0.01:
            return mid
        elif targetvalue > val:
            lower = mid
        elif targetvalue < val:
            upper = mid           
    return best_val        

def grid_search(FARZparameters, targetccoef, targetdega, net, File):
        
    alpharange = np.arange(0,1,0.1)
    gammarange = np.arange(-1,1,0.1) 
       
    min_dif = 1000    
    alphas = []
    alphas2 = []
    gamma = []
    gamma2 = []
    for i in range(len(alpharange)):
        print(i)
        for j in range(len(gammarange)):
            parameters = list(FARZparameters)
            a = alpharange[i]
            g = gammarange[j]
            parameters.append('--alpha')
            parameters.append(str(a))
            parameters.append('--gamma')
            parameters.append(str(g))
            avg_ccoef = 0.0
            avg_dega = 0.0
    
            G,C,FARZsetting = FARZ.FARZ(parameters)
            ccoef = round(nx.average_clustering(G),2)
            dega = round(nx.degree_assortativity_coefficient(G),2)
            dif = abs(ccoef  - targetccoef) + abs(dega - targetdega)
            
            if dif < min_dif:
                min_dif = dif
                bestG = G
                bestC = C
                bestalpha = a
                bestgamma = g
    return bestalpha, bestgamma   
    #         for ind in range(10): # for every parameter set, generate 10 networks    
    #             G,C,FARZsetting = FARZ.FARZ(parameters)
    #             FARZ.write_to_file(G, C, '.', net+"_"+str(ind), 'list2', FARZsetting,True)            
    #             ccoef = round(nx.average_clustering(G),2)
    #             dega = round(nx.degree_assortativity_coefficient(G),2)
    #             dif = abs(ccoef  - targetccoef) + abs(dega - targetdega)
                
    #             if dif < min_dif:
    #                 min_dif = dif
    #                 bestG = G
    #                 bestC = C
    #             avg_ccoef += ccoef
    #             avg_dega += dega
    #         avg_ccoef /= 10
    #         avg_dega /= 10    
    #         alphas.append(a)
    #         alphas2.append(avg_ccoef)
    #         gamma.append(g)
    #         gamma2.append(avg_dega)
    # with open("variables_"+File+".pkl","wb") as f:
    #     pickle.dump((alphas, alphas2, gamma, gamma2),f)
    
    #FARZ.write_to_file(bestG, bestC, '.', net, 'list2', FARZsetting,True)            
            
def loadNetStats(path):
    stats={}
    with open(path) as f:   
        data = json.load(f)
        for key in data:
            stats[key] = data[key]
    return stats

def __main():
    parser = argparse.ArgumentParser(description = 'create the scripts needed to generate networks with FARZ')
    parser.add_argument('net_stats', help='Path to the properties of networks')
    args = parser.parse_args()
    stats={}
    with open('FARZ_parameters','w') as f: 
        for File in os.listdir(args.net_stats):
            parameters = []
            stats = loadNetStats(args.net_stats+"/"+File)
            parameters.append('-n')
            parameters.append(stats['nodes'].encode("utf-8"))
            parameters.append('-m')
            parameters.append(str(round(float(stats['avg_degree'])/2,2)))
            
            parameters.append('-k')
            parameters.append(stats['communities'].encode("utf-8"))
            parameters.append('--beta')
            parameters.append(str(1-float(stats['mixing_parameter'])))
            
            parameters.append('-f')
            parameters.append('list2')
            parameters.append('-o')
            parameters.append("FARZ nets/"+File+"_FARZ")
            #G,C = FARZ.FARZ(parameters)
            #print(parameters)
            
            #best_alpha = binary_search(list(parameters),'alpha',float(stats['avg_clustering_coef']))
            #best_gamma = binary_search(list(parameters),'gamma',float(stats['degree_assortativity']))
            #parameters.append('--alpha')
            #parameters.append(str(best_alpha))
            
            #parameters.append('--gamma')
            #parameters.append(str(best_gamma))
            #FARZ.FARZ(parameters)
            net = "FARZ nets/"+File+"_FARZ"
            bestalpha, bestgamma = grid_search(list(parameters),float(stats['avg_clustering_coef']), float(stats['degree_assortativity']),net,File)
            parameters.append('--alpha')
            parameters.append(str(bestalpha))
            parameters.append('--gamma')
            parameters.append(str(bestgamma))
            for ind in range(10):
                G,C,FARZsetting = FARZ.FARZ(parameters)
                FARZ.write_to_file(G, C, '.', net+"_"+str(ind), 'list2', FARZsetting,True)            

            #print("avg_clustering_coef "+str(round(nx.average_clustering(G),2)))
            f.write(" ".join(str(p) for p in parameters))
            f.write("\n")
            
if __name__ == "__main__":
    __main()
