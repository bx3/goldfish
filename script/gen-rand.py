#!/usr/bin/env python
import sys
import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

np.random.seed(15)

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

def get_dijkstra_dist(G, num_node):
    table = np.zeros((num_node, num_node))
    for i in range(num_node):
        node_dist = nx.shortest_path_length(G, source=i, weight = 'lat')
        for j in range(num_node):
            if float(node_dist[j]) < 0:
                table[i,j] = 0
            else:
                table[i,j] = node_dist[j]
    return table



if len(sys.argv) < 2:
    print('num_node<int>')
    print('num_pub<int>')
    sys.exit(1)

num_node = int(sys.argv[1])
num_pub = int(sys.argv[2])
node_pos = {}
square_len = 250
single_node_dis = 200
for i in range(num_node):
    node_pos[i] = np.random.randint(0, high=square_len, size=2)

all_nodes = [i for i in range(num_node)]
pubs = np.random.choice(all_nodes, num_pub, replace=False)

nodes = []
for i in range(num_node):
    adj = []
    for j in range(num_node):
        dis = math.ceil( math.sqrt((node_pos[i][0]-node_pos[j][0])**2 +
                      (node_pos[i][1]-node_pos[j][1])**2  ))
        adj.append(dis)

    role = 'LURK'
    if i in pubs:
        role = 'PUB'

    node = {
        'id': i,
        'center': None,
        'role': role,
        'x': int(node_pos[i][0]),
        'y': int(node_pos[i][1]),
        'adj': adj 
        }
    nodes.append(node)

summary = {
    'num_node': num_node,
    'square_length': square_len,
    }
setup = {'summary': summary, 'nodes': nodes}

print(json.dumps(setup, indent=4))

# print('dijkstra table')
# for i in range(num_node):
    # text = ["{:4d}".format(int(a)) for a in dijkstra_table[i]]
    # print(text)

# print('dijkstra table')
# for i in range(num_node):
    # text = ["{:4d}".format(int(a)) for a in adj_table[i]]
    # print(text)
# print('diff')
# for i in range(num_node):
    # text = ["{:4d}".format(int(a)) for a in (adj_table[i] - dijkstra_table[i])]
    # print(text)

print()

# write_adj_matrix(G, num_node, outfile)
# pos = nx.spring_layout(G)
# nx.draw(G, pos=pos, with_labels=True)
# edge_labels = nx.get_edge_attributes(G,'lat')
# nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
# plt.savefig('topo.png')



