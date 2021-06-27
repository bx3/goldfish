#!/usr/bin/env python
import sys
import json
from collections import namedtuple
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import random
Node = namedtuple('Node', ['id', 'center_id'])

# complete bipartite connect all of them
def connect_centers(src_c, tgt_c, lat_among_centers, centers, G):
    src_nodes = centers[src_c]
    tgt_nodes = centers[tgt_c]
    for u in src_nodes:
        for v in tgt_nodes:
            lat = int(lat_among_centers+np.random.normal(0,noise_std))
            if lat < 0:
                lat = 0

            G.add_edge(u, v, lat = lat)

def write_adj_matrix(G, num_node, outfile):
    with open(outfile, 'w') as w:
        for i in range(num_node):
            node_dist = nx.shortest_path_length(G, source=i, weight = 'lat')
            for j in range(num_node):
                if float(node_dist[j]) < 0:
                    val = 0
                else:
                    val = float(node_dist[j])
                w.write(str(val) + "  ")
            w.write("\n")
def is_far_enough(dis, c_x, c_y, d_x, d_y):
    d = math.sqrt((c_x-d_x)**2+(c_y-d_y)**2)
    return d > dis

def gen_centers(num_cen, dis, square_len):
    while True:
        centers = []
        for _ in range(num_cen):
            c_x, c_y = np.random.randint(30, high=square_len, size=2)
            centers.append((c_x, c_y))

        # determine
        retry = False 
        for i in range(num_cen):
            c_x, c_y = centers[i]
            for j in range(i+1, num_cen):
                d_x, d_y = centers[j]
                if not is_far_enough(dis, c_x, c_y, d_x, d_y):
                    retry = True
                    break
            if retry:
                break
        if not retry:
            break
    return centers

if len(sys.argv) < 9:
    print('Error need more args')
    print('./gen-datacenter.py use_single<y/n> num_center<int> num_node_per_center<int> num_pub_per_center<int> dis_among_center<float> center_std<int> proc_delay_mean<float> proc_delay_std')
    sys.exit(0)

use_single = sys.argv[1] == 'y'
num_center = int(sys.argv[2])
num_node_per_center = int(sys.argv[3])
num_pub_per_center = int(sys.argv[4])
dis_center = float(sys.argv[5]) # distance among centers
center_std = float(sys.argv[6])
proc_delay_mean = float(sys.argv[7])
proc_delay_std = float(sys.argv[8])


# i.e. use-single = a cluster made of a single node
if use_single:
    num_node = (num_center-1) * num_node_per_center + 1
    centers = defaultdict(list)
    centers[0] = [0]
    for i in range(1, num_node):
        center_id = int((i-1)/num_node_per_center)+1
        centers[center_id].append(i)

    G = nx.Graph()
    # build intracenter lat
    for cid, cluster in centers.items():
        if cid != 0:
            num = len(cluster)
            for i in range(0, num):
                for j in range(i+1, num):
                    lat = int(lat_in_center+np.random.normal(0,noise_std))
                    if lat < 0:
                        lat = 0
                    G.add_edge(cluster[i], cluster[j], lat = lat )

    center_list = list(centers.keys())
    # build intercenter lat
    for i in range(0, num_center):
        src_center = center_list[i]
        for j in range(i+1, num_center):
            tgt_center = center_list[j] 
            connect_centers(src_center, tgt_center, lat_among_centers, centers, G)
else:
    node_pos = {}
    cen_pos = []
    square_len = 250
    node_id = 0
    centers = gen_centers(num_center, dis_center, square_len)
    center_nodes = defaultdict(list)
    node_center = {}

    for i in range(num_center):
        c_x, c_y = centers[i]    
        cen_pos.append((int(c_x), int(c_y)))
        for j in range(num_node_per_center):
            x = c_x + np.random.normal(0, center_std)
            y = c_y + np.random.normal(0, center_std)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            node_pos[node_id] = (x, y)
            center_nodes[i].append(node_id)
            node_center[node_id] = i
            node_id += 1

    num_node = num_node_per_center*num_center
    nodes = []
    
    # assign_pub
    center_pubs = {}
    node_role = {}
    for i in range(num_center):
        pubs = random.choices(center_nodes[i], k=num_pub_per_center)
        center_pubs[i] = pubs

        node_role

    for i in range(num_node):
        adj = []
        for j in range(num_node):
            dis = math.ceil( math.sqrt((node_pos[i][0]-node_pos[j][0])**2 +
                          (node_pos[i][1]-node_pos[j][1])**2  ))
            adj.append(dis)
        role = 'LURK'
        c = node_center[i]
        if i in center_pubs[c]: 
            role = 'PUB'
        proc_delay = np.random.normal(proc_delay_mean, proc_delay_std)


        node = {
            'id': i,
            'center': c,
            'role': role,
            'x': int(node_pos[i][0]),
            'y': int(node_pos[i][1]),
            'proc_delay': proc_delay,
            'adj': adj
            }
        nodes.append(node)

    summary = {
        'num_node': num_node,
        'noise_std': center_std,
        'square_length': square_len,
        'center_pos': cen_pos
        }

    setup = {'summary': summary, 'nodes': nodes}
    print(json.dumps(setup, indent=4)) 
# write_adj_matrix(G, num_node, outfile)
# dump_topo(G, num_node)
