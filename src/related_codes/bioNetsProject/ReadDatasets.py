import networkx as nx
import graph_tool.all as gt
from StatsOfNetworks import *

"""
Set of functions for reading different Biological Networks downloaded from different sources
"""

def readNeuralNet(graph_format ='Networkx'):
    """
        reads the neural network of C. Elegans
        see http://graph-tool.skewed.de/static/doc/collection.html Or
        http://www-personal.umich.edu/~mejn/netdata/
        
        :input graph_format: the data structure format of the graph, either  'Networkx' or 'Graph-tool'
        
        :return g: a directed, weighted network in the specified format and also the name of the network
        
    """
    
    if graph_format== 'Networkx':
        g = nx.read_gml('../datasets/neural/celegansneural.gml')
    elif graph_format == 'Graph-tool':
        g = gt.collection.data["celegansneural"]
    else: 
        raise ValueError(graph_format + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
    return {'graph':g, 'name':'celegans', 'type':'Neural', 'graph_format':graph_format}

# could also include monkeys interactions: http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm#wolf
# or baboon grooming http://vlado.fmf.uni-lj.si/pub/networks/data/GBM/baboons.htm
def readSpeciesSocialInteractionNet(graph_format ='Networkx'):
    """
        reads the social network of frequent associations between 62 dolphins in a community living off Doubtful Sound, New Zealand
        see http://graph-tool.skewed.de/static/doc/collection.html Or
        http://www-personal.umich.edu/~mejn/netdata/

        :input graph_format: the data structure format of the graph, either  'Networkx' or 'Graph-tool'
        
        :return g: an undirected network in the specified format
    """
    
    if graph_format== 'Networkx':
        g = nx.read_gml('../datasets/speciesInteraction/dolphins.gml')
    elif graph_format == 'Graph-tool':
        g = gt.collection.data["dolphins"]
    else: 
        raise ValueError(graph_format + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
    return {'graph':g, 'name':'dolphins', 'type':'speciesInteraction', 'graph_format':graph_format}


# could also include these http://vlado.fmf.uni-lj.si/pub/networks/data/bio/foodweb/foodweb.htm
def readFoodWebNet(graph_format ='Networkx'):
    """
        reads the Plant and mammal food web from the Serengeti savanna ecosystem in Tanzania
        see http://graph-tool.skewed.de/static/doc/collection.html 

        :input graph_format: the data structure format of the graph, either  'Networkx' or 'Graph-tool'
        
        :return g: a directed network in the specified format
    """
    
    if graph_format== 'Networkx':
        g = nx.read_graphml('../datasets/foodweb/serengeti-foodweb.graphml')
    elif graph_format == 'Graph-tool':
        g = gt.collection.data["serengeti-foodweb"]
        #print gt.collection.get_data_path("serengeti-foodweb")
    else: 
        raise ValueError(graph_format + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
    return {'graph':g, 'name':'Serengeti', 'type':'foodweb', 'graph_format':graph_format}

def readProtienProteinInteractionNet(graph_format ='Networkx'):
    """
        reads the Protein-protein interaction network in budding yeast
        see http://vlado.fmf.uni-lj.si/pub/networks/data/bio/Yeast/Yeast.htm 

        :input graph_format: the data structure format of the graph, either  'Networkx' or 'Graph-tool'
        
        :return g: a directed network in the specified format
    """
    path = '../datasets/protein/yeast/YeastS'
    if graph_format== 'Networkx':
        g = nx.read_pajek(path+".net")
       # nx.write_gml(g, path+'.gml')  #saving into GML since Graph-tool can not load pajak files
    elif graph_format == 'Graph-tool':
        g = gt.load_graph(path+'.gml')
    else: 
        raise ValueError(graph_format + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
    return  {'graph':g, 'name':'yeast', 'type':'PPI', 'graph_format':graph_format}
'''
#from http://vlado.fmf.uni-lj.si/pub/networks/data/bio/Yeast/Yeast.htm
g = nx.read_pajek('../datasets/Protien Interaction/yeast/YeastS.net')
#from http://www3.nd.edu/~networks/resources.htm
g2 = nx.read_edgelist('../datasets/Protien Interaction/yeast2/bo.dat')
printGraphStats(g)
printGraphStats(g2)

#>> UnDirected  Graph loaded with  2361 Nodes and  7182 Edges 
#>> Density of the Graph is  0.0025779079534  and it has  101  components
#>> UnDirected  Graph loaded with  1870 Nodes and  2277 Edges 
#>> Density of the Graph is  0.00130299310736  and it has  173  components

#It seems that these two graphs (from different sources) are although represent the same network, are not equal
#I use the bigger network
'''

# Also to include others from http://www3.nd.edu/~networks/resources/metabolic/index.html
# Name of organisms : http://www3.nd.edu/~networks/resources/metabolic/supply.htm
# SC:yeast (Saccharomyces cerevisiae), CE:C.elegans 
def readMetabolicNet(graph_format ='Networkx'):
    """
        reads the metabolic network of C.elegans
        see http://deim.urv.cat/~aarenas/data/welcome.htm

        :input graph_format: the data structure format of the graph, either  'Networkx' or 'Graph-tool'
        
        :return g: a directed network in the specified format
    """    
    path= '../datasets/metabolic/Celegans/celegans_metabolic'
    if graph_format== 'Networkx':
        g = nx.read_pajek(path+'.net')
       # nx.write_gml(g, path+'.gml')  #saving into GML since Graph-tool can not load pajak files
    elif graph_format == 'Graph-tool':
        g = gt.load_graph(path+'.gml')
    else: 
        raise ValueError(graph_format + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
    return  {'graph':g, 'name':'celegans', 'type':'metabolic', 'graph_format':graph_format}

'''
g = nx.read_gml('../datasets/metabolic network/Celegans/celegans.gml')
g2 = nx.read_pajek('../datasets/metabolic network/Celegans/celegans_metabolic.net')
g3 = nx.DiGraph()
g3 = nx.read_edgelist('../datasets/metabolic network/CE.dat', create_using=g3)
printGraphStats(g)
printGraphStats(g2)
printGraphStats(g3)

#>> Directed  Graph loaded with  306 Nodes and  2345 Edges 
#>> Density of the Graph is  0.0251258973535  and it has  10  components
#>> Directed  Graph loaded with  453 Nodes and  4596 Edges 
#>> Density of the Graph is  0.0224462286819  and it has  1  components
#>> UnDirected  Graph loaded with  1173 Nodes and  2842 Edges 
#>> Density of the Graph is  0.00413455187684  and it has  1  components

#Again it seems that these three graphs (from different sources although represent the same network), are not similar
#I use the second network
'''

# email-Enron, 

# could also add other organisms from http://www3.nd.edu/~networks/resources/cellular/index.html
# Name of organisms : http://www3.nd.edu/~networks/resources/metabolic/supply.htm
# main entry: http://www3.nd.edu/~networks/resources.htm
def readCellularNet(graph_format ='Networkx'):
    """
        reads the  the whole cellular network of C.elegans
        see http://www3.nd.edu/~networks/resources.htm

        :input graph_format: the data structure format of the graph, either  'Networkx' or 'Graph-tool'
        
        :return g: a directed network in the specified format
    """    
    path= '../datasets/celluar/CE'
    if graph_format== 'Networkx':
        g = nx.DiGraph()
     #   printGraphStats(g)

        g = nx.read_edgelist(path+'.dat', create_using=g)
      #  printGraphStats(g)

     #   nx.write_gml(g, path+'.gml')  #saving into GML since Graph-tool can not load pajak files
    elif graph_format == 'Graph-tool':
        g = gt.load_graph(path+'.gml')
    else: 
        raise ValueError(graph_format + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
    return {'graph':g, 'name':'celegans', 'type':'cellular', 'graph_format':graph_format}



def readSynthNet(graph_format ='Networkx'):
    """
        reads the  the whole cellular network of C.elegans
        see http://www3.nd.edu/~networks/resources.htm

        :input graph_format: the data structure format of the graph, either  'Networkx' or 'Graph-tool'
        
        :return g: a directed network in the specified format
    """    
    print  graph_format
    path= '../datasets/test2.csv'
    if graph_format== 'Networkx':
        g = nx.DiGraph()
     #   printGraphStats(g)

        g = nx.read_edgelist(path,delimiter=';', create_using=g)
       # printGraphStats(g)

        nx.write_gml(g, path+'.gml')  #saving into GML since Graph-tool can not load pajak files
    elif graph_format == 'Graph-tool':
        g = gt.load_graph(path+'.gml')
    else: 
        raise ValueError(graph_format + 'is invalid, format must be either \'Networkx\' or \'Graph-tool\'')
    return {'graph':g, 'name':'celegans', 'type':'cellular', 'graph_format':graph_format}

# More datasets http://sbcny.org/data.htm
#http://tuvalu.santafe.edu/~aaronc/powerlaws/data.htm
#http://tuvalu.santafe.edu/~aaronc/powerlaws/bins/
#body mass information for all mammals: http://www.esapubs.org/archive/ecol/E084/094/#data
#The human disease network: http://www.pnas.org/content/104/21/8685.abstract?tab=ds