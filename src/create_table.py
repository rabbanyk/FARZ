from __future__ import print_function
import os
import argparse
import networkx as nx
import json
import decimal

def loadNetStats(path):
    stats={}
    with open(path) as f:   
        data = json.load(f)
        for key in data:
            stats[key] = data[key]
    return stats

    return stats
def __main():
    parser = argparse.ArgumentParser(description = 'create a latex table based on the properties of networks')
    parser.add_argument('net_stats', help='Path to the properties of networks')
    args = parser.parse_args()
    stats={}
    parameters = ['nodes','edges','communities','diameter','average_shortest_path','degree_assortativity','avg_clustering_coef']
    print("\\begin{table}[]")
    print("\centering")
    print("\caption{My caption}")
    print("\label{my-label}")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|}")
    print("\hline")
    prop_list = ['nodes', 'edges','communities','diameter', 'average\_shortest\_path', 'degree\_assortativity','avg\_clustering\_coef']
    for prop in prop_list:
        print(" & \\rotatebox{90}{ "+prop+" }", end='')
    print("  \\\\ \hline")
    for File in sorted(os.listdir(args.net_stats)):
        stats[File] = loadNetStats(args.net_stats+"/"+File)
        print(File.replace('_','-'), end='')
        for p in parameters:
            if "avg" not in stats[File]:
                if isinstance(stats[File][p], float):
                    print(" & "+str(round(stats[File][p],2)), end='') 
                else:
                    print(" & "+str(stats[File][p]), end='') 
            else:
                if isinstance(stats[File]["avg"][p], float):
                    avg = decimal.Decimal(str(round(stats[File]["avg"][p],2))).normalize()
                    std = decimal.Decimal(str(round(stats[File]["std"][p],2))).normalize()
                    print(" & "+str(avg)+"("+str(std)+")", end='') 
                else:
                    print(" & -", end='')
        print("  \\\\ \hline")
    #print("& nodes & edges & communities & diameter & average\_shortest\_path & degree\_assortativity & avg\_clustering\_coef \\\\ \hline")
    # for File in sorted(os.listdir(args.net_stats)):
    #     stats[File] = loadNetStats(args.net_stats+"/"+File)
    #     print(" & \\rotatebox{90}{ "+File.replace('_','-')+" }", end='')
    #     #print(File+"& "+stats["nodes"]+" & "+stats["edges"]+" & "+stats["communities"]+" & "+stats["diameter"]+" & "+stats["average_shortest_path"]+" & "+stats["degree_assortativity"]+" & "+stats["avg_clustering_coef"]+" \\\\ \hline")
    # print("  \\\\ \hline")   
    # for p in parameters:
    #     print(p.replace('_',' '), end='')
    #     for File in sorted(os.listdir(args.net_stats)):
    #         print(" & "+stats[File][p], end='')
    #     print("  \\\\ \hline")    
    print("\end{tabular}")
    print("\end{table}")    
if __name__ == "__main__":
    __main()
