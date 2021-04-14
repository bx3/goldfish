#!/usr/bin/env python
import sys
import json
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

np.random.seed(17)

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
    print('num_node<int>  output')
    sys.exit(1)

num_node = int(sys.argv[1])
noise_std = 10
node_pos = {}
square_len = 250
single_node_dis = 200
for i in range(num_node):
    node_pos[i] = np.random.randint(0, high=square_len, size=2)

adj_table = np.zeros((num_node, num_node))
for i in range(num_node):
    for j in range(i+1, num_node):
        if i == 0:
            d = int(single_node_dis + np.random.normal(0,noise_std))
            adj_table[i,j] = d
            adj_table[j,i] = d 
        else:
            d = math.ceil(math.sqrt((node_pos[i][0]-node_pos[j][0])**2+(node_pos[i][1]-node_pos[j][1])**2))
            adj_table[i,j] = d
            adj_table[j,i] = d
# for i in range(num_node):
    # text = ["{:4d}".format(int(a)) for a in adj_table[i]]
    # print(text)

G = nx.Graph()
for i in range(num_node):
    for j in range(num_node):
        G.add_edge(i, j, lat = adj_table[i,j])

dijkstra_table = get_dijkstra_dist(G, num_node)
# print(dijkstra_table[0])

# print(dijkstra_table[0] - adj_table[0])


nodes = []
for i in range(num_node):
    node = {
        'id': i,
        'x': int(node_pos[i][0]),
        'y': int(node_pos[i][1]),
        'adj': list(dijkstra_table[i,:])
        }
    nodes.append(node)

summary = {
    'num_node': num_node,
    'noise_std': noise_std,
    'square_length': square_len,
    'single_node_dis': single_node_dis
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



