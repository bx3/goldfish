import numpy as np
import json
import networkx as nx
from collections import defaultdict
import random

def get_num_pub_node(json_file):
    num_pub = 0
    num_node = None
    pubs = []
    with open(json_file) as config:
        data = json.load(config)
        nodes = data['nodes']
        summary = data['summary']
        num_node = summary['num_node']
        role = {}

        for node in nodes:
            if node["role"] == 'PUB':
                num_pub += 1
                pubs.append(node["id"])
    return num_pub, num_node, pubs



def load_network(json_file):
    with open(json_file) as config:
        data = json.load(config)
        nodes = data['nodes']
        summary = data['summary']
        num_node = summary['num_node']

        ld = {}
        proc_delay = {}
        loc = {}
        role = {}
        for node in nodes:
            i = node['id']
            adj = node['adj']
            x = node['x']
            y = node['y']
            proc_delay[i] = float(node['proc_delay'])
            loc[i] = (x,y)
            ld[i] = adj
            role[i] = node["role"]
        return loc, ld , role, proc_delay

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

def generate_random_outs_conns_fix_seed(out_lim, in_lim, num_node, single_cluster, seed):
    outs_conns = defaultdict(list) 

    nodes = [i for i in range(num_node)]
    in_counts = {i:0 for i in range(num_node)}
    pools = nodes.copy()
    rng = np.random.RandomState(seed)
    for _ in range(out_lim):
        random.shuffle(nodes)
        for i in nodes:
            w = rng.choice(pools)
            while ( w in outs_conns[i] or
                    w == i
                    # in_counts[w] >= in_lim or
                    # i in outs_conns[w] # allow two bidictioncal arrows
                    ):
                w = rng.choice(pools)
            outs_conns[i].append(w)

    return outs_conns

def generate_random_outs_conns_with_oracle(out_lim, num_node, oracle):
    nodes = [i for i in range(num_node)]
    pools = nodes.copy()
    outs_conns = defaultdict(list)
    for _ in range(out_lim):
        random.shuffle(nodes)
        for i in nodes:
            w = np.random.choice(pools)
            while w == i or w in outs_conns[i] or len(oracle.can_i_connect(i, [w])) != 0:
                w = np.random.choice(pools)
            oracle.update(i, [], [w], [])
            outs_conns[i].append(w)
    return outs_conns


def generate_random_outs_conns(out_lim, in_lim, num_node, single_cluster):
    outs_conns = defaultdict(list) 

    nodes = [i for i in range(num_node)]
    in_counts = {i:0 for i in range(num_node)}
    if single_cluster == 'y':
        nodes.remove(0)
    pools = nodes.copy()

    for _ in range(out_lim):
        random.shuffle(nodes)
        for i in nodes:
            w = np.random.choice(pools)
            while ( w in outs_conns[i] or
                    w == i
                    # in_counts[w] >= in_lim or
                    # i in outs_conns[w] # allow two bidictioncal arrows
                    ):
                w = np.random.choice(pools)
            outs_conns[i].append(w)

    if single_cluster == 'y':
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

