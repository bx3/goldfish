import sys
import os
import math
import json
import matplotlib.pyplot as plt
from plot_topo import plot_graph
from plot_topo import plot_topology
from plot_utility import parse_adapt

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

if len(sys.argv) < 3:
    print('need exp dir, epochs')
    sys.exit(1)

exp_dir = sys.argv[1]
epochs = [int(e) for e in sys.argv[2:]]

topo_json = os.path.join(exp_dir, 'topo.json')
graph_dir = os.path.join(exp_dir, 'graphs')
stars = parse_adapt(os.path.join(exp_dir, 'adapts'))
num_pub, num_node, pubs = get_num_pub_node(topo_json)

num_plot = len(epochs) + 1

num_row = min(2, num_plot)
num_col = int(math.ceil(num_plot/float(num_row)))

fig, axs = plt.subplots(ncols=num_col, nrows=num_row, figsize=(10*num_col,10*num_row))
if num_row == 1 and num_col == 1:
    axs = [axs]
else:
    axs = axs.flatten()
plot_topology(topo_json, axs[0], stars)

for i,e in enumerate(epochs):
    graph_json = os.path.join(graph_dir, 'epoch'+str(e)+'.json')
    plot_graph(topo_json, graph_json, axs[i+1], stars, pubs)

plt.savefig('graphs')
