# FARZ 
## Benchmarks for Community Detection Algorithms

FARZ is generator/simulator for networks with built-in community structure. 
It creates graphs/networks with community labels, which can be used for evaluating community detection algorithms.

### Generator Parameters
* main parameter
   + `n`: number of nodes
   + `m`: half the average degree of nodes
   + `k`: number of communities
* control parameters
   + `beta`: the strength of community structure, i.e. the probability of edges to be formed within communities, default (0.8)
   + `alpha`: the strength of common neighbor's effect on edge formation edges, default (0.5)
   + `gamma`: the strength of degree similarity effect on edge formation, default (0.5), can be negative for networks with negative degree correlation
* overlap parameters
   + `r`: the number of communities each node can belong to, default (1)
   + `q`: the probability of a node belonging to the multiple communities, default (0.5)
* config parameters
   + `phi`: the constant added to all community sizes, higher number makes the communities more balanced in size, default (1) which results in power law distribution for community sizes
   + `epsilon`: the probability of noisy/random edges, default (0.0000001)
   + `t`: the probability of also connecting to the neighbors of a node each nodes connects to. The default value is (0), but could be increase to a small number to achieve higher clustering coefficient. 
 
### How to run
The source code is in Pyhton 2.7. 
You can generate FARZ benchmarks from FARZ.py in src. 
See below for examples.

### Examples
	+ example 1: generate a network with 1000 nodes and about 5x1000 edges (m=5), with 4 communities, where 90% of edges fall within communities (beta=0.9)
   	` python FARZ.py -n 1000 -m 5 -k 4 --beta 0.9`
    + example 2: generate a network with properties of example 1, where alpha = 0.2 and gamma = -0.8
    ` python FARZ.py -n 1000 -m 5 -k 4 --beta 0.9 --alpha 0.2 --gamma -0.8 `
    + example 3: generate 10 sample networks with properties of example 1 and save them into ./data
    ` python FARZ.py --path ./data -s 10 -n 1000 -m 5 -k 4 --beta 0.9`
    + example 4: repeat example 2, for beta that varies from 0.5 to 1 with 0.05 increments'
    ` python FARZ.py --path ./data -s 10 -v beta -c [0.5,1,0.05] -n 1000 -m 5 -k 4 `
    + example 5: generate overlapping communities, where each node belongs to at most 3 communities and the portion of overlapping nodes varies'
    ` python FARZ.py -r 3 -v q --path ./datavrq -s 5 --format list`

### Support or Contact
Reihaneh Rabbany, rabbanyk@ualberta.ca
