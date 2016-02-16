# FARZ 
## Benchmarks for Community Detection Algorithms

FARZ is generator/simulator for networks with built-in community structure. 
It creates graphs/networks with community labels, which can be used for evaluating community detection algorithms.

### Generator Parameters
* main parameter
   + `n`: number of nodes
   + `k`: number of communities
   + `m`: half the average degree of nodes
* control parameters
   + `beta`: the strength of community structure, i.e. the probability of edges to be formed within communities, default (0.8)
   + `alpha`: the strength of common neighbor's effect on edge formation edges, default (0.5)
   + `gamma`: the strength of degree similarity effect on edge formation, default (0.5), can be negative for networks with negative degree correlation
* config parameters
   + `phi`: the constant added to all community sizes, higher number makes the communities more balanced in size, default (1) which results in power law distribution for community sizes
   + `o`: the number of communities each node can belong to, default (1) 
   + `epsilon`: the probability of noisy/random edges, default (0.0000001)
   + `b`: the probability of also connecting to the neighbors of a node each nodes connects to. The default value is (0), but could be increase to a small number to achieve higher clustering coefficient. 
 
### How to run
The source code is in Pyhton 2.7. 
You can generate FARZ benchmarks from FARZ.py. See below for examples.

### Examples
`params = {"n":1000, "k":4, "m":4, "alpha":0.5, "beta":0.8, "gamma":0.5}`

* Use FARZ directly
   + `FARZ.realize(**params)`
* Use batch generator
   + Generate a dataset of networks with `beta` varying from 0.5 to 1 with 0.05 steps, for each beta generate 10 networks, and write the networks in gml format.

    `generate('beta', arange=(0.5,1,0.05), repeat =10, path='./vbeta', format = 'gml', params=params)`


### Reference 
[Paper on Arxiv]()

### Support or Contact
Reihaneh Rabbany (@rabbanyk), rabbanyk@ualberta.ca
