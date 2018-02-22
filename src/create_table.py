import os
import argparse
import networkx as nx
import json
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
    print("\\begin{table}[]")
    print("\centering")
    print("\caption{My caption}")
    print("\label{my-label}")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|c|}")
    print("\hline")
    print("& nodes & edges & communities & diameter & average\_shortest\_path & degree\_assortativity & avg\_clustering\_coef \\\\ \hline")
    for File in os.listdir(args.net_stats):
        stats = loadNetStats(args.net_stats+"/"+File)
        print(File+"& "+stats["nodes"]+" & "+stats["edges"]+" & "+stats["communities"]+" & "+stats["diameter"]+" & "+stats["average_shortest_path"]+" & "+stats["degree_assortativity"]+" & "+stats["avg_clustering_coef"]+" \\\\ \hline")
    print("\end{tabular}")
    print("\end{table}")    
if __name__ == "__main__":
    __main()
