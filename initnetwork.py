#!/usr/bin/env python
import networkx as nx
from math import radians, cos, sin, asin, sqrt
import numpy as np
import matplotlib.pyplot as plt
import random
import data
import sys
import math
import readfiles
from collections import defaultdict
import config
from config import *

def construct_graph_by_outs_conns(outs_conns, nd, ld):
    G = nx.Graph()
    for i, nodes in outs_conns.items():
        for u in nodes:
            delay = ld[i][u] + nd[i]/2 + nd[u]/2
            assert(i != u)
            G.add_edge(i, u, weight=delay)
    return G


def construct_graph(nodes, ld):
    num_nodes = len(nodes)
    G = nx.Graph()
    for i, node in nodes.items():
        for u in node.outs:
            delay = ld[i][u] + node.node_delay/2 + nodes[u].node_delay/2
            assert(i != u)
            G.add_edge(i, u, weight=delay)
    return G

def construct_graph_revise(nodes, ld):
    pass

def reduce_link_latency(num_node, num_low_latency, ld):
    all_nodes = [i for i in range(num_node)]
    random.shuffle(all_nodes)
    all_nodes = list(all_nodes)
    low_lats = all_nodes[:num_low_latency]

    for i in low_lats:
        for j in low_lats:
            if i != j:
                ld[i][j] *= config.reduce_link_ratio 

def generate_random_outs_conns(out_lim, in_lim, num_node, single_cluster):
    outs_conns = defaultdict(list) 

    nodes = [i for i in range(num_node)]
    in_counts = {i:0 for i in range(num_node)}
    if single_cluster == 'y':
        nodes.remove(0)

    for _ in range(out_lim):
        random.shuffle(nodes)
        for i in nodes:
            w = np.random.choice(nodes)
            while ( w in outs_conns[i] or
                    w == i or
                    # in_counts[w] >= in_lim or
                    i in outs_conns[w]
                    ):
                w = np.random.choice(nodes)
            outs_conns[i].append(w)
    outs_conns[0] = np.random.permutation([i for i in range(1, num_node)])[:out_lim] 
    return outs_conns

def gen_dc_connected(num_node, num_cluster, num_conn, single_cluster):
    outs_conns = defaultdict(list) 
    clusters = defaultdict(list)
    node_to_cluster = {}

    # data structure setup
    num_nodes_per_cluster = int((num_node-1)/(num_cluster-1))
    clusters[0] = [0]
    for i in range(1, num_cluster):
        for j in range((i-1)* num_nodes_per_cluster+1, (i)* num_nodes_per_cluster+1):
            clusters[i].append(j)
            node_to_cluster[j] = i
    node_to_cluster[0] = 0

    nodes = list(node_to_cluster.keys())
    nodes.remove(0)
    print(clusters)
    print(node_to_cluster)
    for c in range(1, num_cluster):
        c_nodes = clusters[c]
        # print('c_nodes', c_nodes)
        o_nodes = list(set(nodes).difference(set(c_nodes)))
        # print('o_nodes', o_nodes)
        for j in c_nodes:
            num_intra_edge = num_conn 
            if np.random.rand()> 0.5:
                num_intra_edge -= 1
                for _ in range(num_intra_edge):
                    k = np.random.choice(c_nodes) 
                    while j == k or  k in outs_conns[j]: #(j in outs_conns[k]) or:
                        k = np.random.choice(c_nodes)         
                    outs_conns[j].append(k)
                # inter-datacenter edge
                
                k = np.random.choice(o_nodes)
                while j == k or (j in outs_conns[k]) or k in outs_conns[j]:
                    k = np.random.choice(o_nodes)         
                outs_conns[j].append(k)
            else:
                for _ in range(num_intra_edge):
                    k = np.random.choice(c_nodes) 
                    while j == k or  k in outs_conns[j]: #(j in outs_conns[k]) or:
                        k = np.random.choice(c_nodes)         
                    outs_conns[j].append(k)

    outs_conns[0] = np.random.permutation([i for i in range(1, num_node)])[:num_conn] 
    # print(outs_conns)
    return outs_conns

def generate_cluster_outs_conns(out_lim, in_lim, num_node, num_cluster, isolate_cluster, single_cluster):
    outs_conns = defaultdict(list) 
    order = [i for i in range(num_node)]
    in_counts = {i:0 for i in range(num_node)}
    clusters = defaultdict(list)
    node_to_cluster = {}
    if not single_cluster:
        num_nodes_per_cluster = int(num_node/num_cluster)
        for i in range(num_cluster):
            for j in range(i* num_nodes_per_cluster, (i+1)* num_nodes_per_cluster):
                clusters[i].append(j)
                node_to_cluster[j] = i
    else:
        num_nodes_per_cluster = int((num_node-1)/(num_cluster-1))
        clusters[0] = [0]
        for i in range(1, num_cluster):
            for j in range((i-1)* num_nodes_per_cluster+1, (i)* num_nodes_per_cluster+1):
                clusters[i].append(j)
                node_to_cluster[j] = i

    others = []
    for i in range(num_cluster):
        if i != isolate_cluster:
            others += clusters[i]

    for i in range(num_cluster):
        for j in clusters[i]:
            if i != isolate_cluster:
                cands = others.copy()
                cands.remove(j)
                outs_conns[j] = np.random.permutation(cands)[:out_lim] 
            else:
                cands = clusters[i].copy()
                cands.remove(j)
                outs_conns[j] = np.random.permutation(cands)[:out_lim] 
    outs_conns[0] = np.random.permutation([i for i in range(1, num_node)])[:out_lim] 
    return outs_conns





# Generate the initial graph with all the lawful users
def GenerateInitialGraph():
    G = nx.Graph()
    data_file = open(config.data_file, 'r', errors='replace')
    line = data_file.readline()
    lines = line.split('],')
    k=0
    for i in range(9559):
        a = lines[i].split(',')
        if(data.con[a[7]]!="Null"):
            G.add_node(k,country=a[7], cluster=data.con[a[7]])
            k=k+1
    data_file.close()
    return(G)

# Generate nodes' processing delay
def GenerateInitialDelay(num_node):
    delay=[0 for i in range(num_node)]
    for i in range(num_node):
        buff= np.random.normal(config.node_delay_mean, config.node_delay_std)
        delay[i]=round(buff,6)
    return(delay)

# Generate the random neighbor connection
def GenerateOutNeighbor(len_of_neigh,IncomingLimit, num_node):
    OutNeighbor= np.zeros([num_node,len_of_neigh], dtype=np.int32)
    IncomingNeighbor=np.zeros(num_node, dtype=np.int32)
    for i in range(num_node):
        for j in range(len_of_neigh):
            OutNeighbor[i][j]=np.random.randint(num_node)
            out_peer = int(OutNeighbor[i][j])
            while( (out_peer in OutNeighbor[i][:j]) or 
                   (out_peer==i) or 
                   IncomingNeighbor[out_peer]>=IncomingLimit[out_peer] 
                  or (out_peer < i and i in OutNeighbor[out_peer] )):
                OutNeighbor[i][j]=np.random.randint(num_node)
                out_peer = int(OutNeighbor[i][j])
            IncomingNeighbor[out_peer]=IncomingNeighbor[out_peer]+1
    return(OutNeighbor,IncomingNeighbor)
    
    
# NeighborSets, contains connectionconuts and  all neighbor ids (including the incomings)
def GenerateInitialConnection(OutNeighbor,len_of_neigh, num_node):
    NeighborSets = np.zeros([num_node, config.in_lim+1+len_of_neigh ]) #225+len_of_neigh]) #225+len_of_neigh 1001
    for i in range(num_node):
        NeighborSets[i][0]=8
        for j in range(len_of_neigh):
            NeighborSets[i][1+j]=int(OutNeighbor[i][j])

    for i in range(num_node):
        for j in range(len_of_neigh):
            peer = int(OutNeighbor[i][j])
            peer_conn_count = int(NeighborSets[peer][0])
            # print("node",i, "peer",  peer, "peer_conn_count", peer_conn_count)
            # TODO is it not a bug?
            if i not in NeighborSets[peer][1:peer_conn_count+1]:
                NeighborSets[peer][peer_conn_count+1]=i
                NeighborSets[peer][0] += 1
    return(NeighborSets)

# if the block size is large enough, get linkdelays by the bandwidth
def DelayByBandwidth(NeighborSets,bandwidth, num_node):
    weight_table=np.zeros([num_node,num_node])
    for i in range(num_node):
        for j in range(len_of_neigh):
            if i != j :
                weight_table[i][int(OutNeighbor[i][j])] = 8 / min(bandwidth[i]/int(NeighborSets[i][0]) , bandwidth[int(OutNeighbor[i][j])]/int(NeighborSets[int(OutNeighbor[i][j])][0]))
    return(weight_table)
    
# Build graph edges
def BuildNeighborConnection(G,OutNeighbor,LinkDelay,delay,len_of_neigh, num_node):
    for i in range(num_node):
        for j in range(len_of_neigh):
            # plus half of both two sides' processing delay so that the node's processing delay can also be considered during shortest path searching
            G.add_edge(i,OutNeighbor[i][j],weight=LinkDelay[i][int(OutNeighbor[i][j])]+delay[i]/2+delay[int(OutNeighbor[i][j])]/2)
    return(G)

def InitBandWidth(num_node):
    bandwidth=np.zeros(num_node)
    for i in range(num_node):
        if (random.random()<0.33):
            bandwidth[i]=50
        else:
            bandwidth[i]=12.5
    return(bandwidth)

def InitIncomLimit(num_node):
    IncomingLimit=np.zeros(num_node)
    for i in range(num_node):
        #IncomingLimit[i]=min(int(bandwidth[i]*1.5),200)
        IncomingLimit[i]=config.in_lim
    return(IncomingLimit)


def GenerateInitialNetwork( NetworkType, num_node, subcommand, out_lim):
    bandwidth=InitBandWidth(num_node)
    IncomingLimit   =   InitIncomLimit(num_node)
    # G               =   GenerateInitialGraph()
    NodeDelay       =   GenerateInitialDelay(num_node)
    if subcommand == 'run-mf' or subcommand == 'run-2hop':
        [OutNeighbor,IncomingNeighbor]     =   GenerateOutNeighbor(out_lim,IncomingLimit, num_node)
    else:
        OutNeighbor = None
        IncomingNeighbor = None

    NeighborSets = None

    [LinkDelay,NodeHash,NodeDelay] = readfiles.Read(NodeDelay, NetworkType, num_node)

    if config.is_load_conn:
        print("\033[91m" + 'Use. Preset conn'+ "\033[0m")
        out_conns = {}
        with open(config.conn_path) as f:
            for line in f:
                tokens = line.strip().split(' ')
                out_conns[int(tokens[0])] = [int(i) for i in tokens[1:]]
        OutNeighbor = out_conns
    else:
        print("\033[93m" + 'Not use. Preset conn'+ "\033[0m")

    return(NodeDelay,NodeHash,LinkDelay,NeighborSets,IncomingLimit,OutNeighbor,IncomingNeighbor,bandwidth)
 
# Update graph by the latest neighbor connections
def UpdateNetwork(G,OutNeighbor,LinkDelay,NodeDelay,len_of_neigh,NeighborSets,bandwidth):
    NeighborSets=   GenerateInitialConnection(OutNeighbor,len_of_neigh)
    #LinkDelay=   initnetwork.DelayByBandwidth(NeighborSets,bandwidth)
    G           =   BuildNeighborConnection(G,OutNeighbor,LinkDelay,NodeDelay,len_of_neigh, num_node)
    return(NeighborSets,LinkDelay,G)

