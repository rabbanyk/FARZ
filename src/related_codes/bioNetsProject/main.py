from ReadDatasets import *
from VisualizeNetworks import *
from StatsOfNetworks import *
from AnalyzeNetworks import *


formats = ['Networkx','Graph-tool']


#------------------------------ Part I & II : reading datasets and computing their characteristics ----------------------------------------
biologicalGraphs=[]

# load different biological networks
print 'Reading networks'
for f in formats:
    arg = {'graph_format':f}
    biologicalGraphs.append([readSynthNet(**arg) ])
    
    '''biologicalGraphs.append([readNeuralNet(**arg) ])
     , readSpeciesSocialInteractionNet(**arg)
    , readFoodWebNet(**arg) 
    , readProtienProteinInteractionNet(**arg)
    , readMetabolicNet(**arg)
    , readCellularNet(**arg)])'''

g0 = readSynthNet(**{'graph_format':'Networkx'})
g1 = readSynthNet(**{'graph_format':'Graph-tool'})
plotDegreeSeq(g0['graph'],g1['graph'])
compareDifferentGraphLayoutsInNetworkX(g0['graph'])
compareDifferentGraphLayoutsInGraphtool(g1['graph'])
# Compute basic characteristics of loaded datasets
print 'Computing basic characteristics networks'

for gs in biologicalGraphs:
    for g in gs: printGraphStats(g) 


#------------------------------ Part III : analyzing networks ----------------------------------------
# Partition loaded datasets
print 'finding communities in the networks'
for gs in biologicalGraphs:
    for g in gs: findCommunities(g) 

# Ranking nodes using centrality analysis
'''print 'centrality analysis'
for gs in biologicalGraphs:
    for g in gs: computeCentralities(g) 
'''

