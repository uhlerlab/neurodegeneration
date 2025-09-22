import os

import numpy as np




import networkx as nx



def modularity_perCommunity(partition,G,weight="weight",resolution=1):
    ###modified from networkx modularity function###
    out_degree = in_degree = dict(G.degree(weight=weight))
    deg_sum = sum(out_degree.values())
    m = deg_sum / 2
    norm = 1 / deg_sum**2
    
    mods=np.zeros(len(partition))
    for commIdx in range(len(partition)):
        comm=partition[commIdx]
        comm = set(comm)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum =  out_degree_sum

        mods[commIdx]= L_c / m - resolution * out_degree_sum * in_degree_sum * norm
    return mods

