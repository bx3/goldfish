#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import os
import json
import numpy as np
import matplotlib.patches as mpatches
import math
import matplotlib.ticker as ticker
import statistics
from collections import defaultdict
from plot_utility import plot_figure

if len(sys.argv) < 4:
    print("Require. num_node num_epoch prefix")
    sys.exit(1)

def parse_node_epoch(msg_file, time_file):
    msg_data = []
    if not os.path.exists(msg_file) or not os.path.exists(time_file):
        return {}
    with open(msg_file) as f:
        msg_data = json.load(f)
        # print(len(msg_data), msg_data)
    time_data = []
    with open(time_file) as f:
        time_data = json.load(f)
        # print(len(time_data), time_data)

    assert(len(time_data)==len(msg_data))
    num_msg = len(time_data)

    pub_times = defaultdict(list) # key is publisher, value is time list 
    for i in range(num_msg):
        msg = msg_data[i]
        pub_t = float(msg['PubTime'])
        peers_t = []
        for mem in time_data[i]:
            if mem[1] != "None":
                 peers_t.append((float(mem[1]) - pub_t)/1e4)
        pub_times[msg['PubID']].append(min(peers_t))
    pub_median = {}
    for pub, times in pub_times.items():
        pub_median[pub] = statistics.median(times)
    return pub_median

def plot_time(epochs_data, num_node, stars, x):
    epochs_lats = {}
    max_y = 0 
    min_y = 1e8
    for e, nodes in epochs.items():
        latencies = []
        for i, lats in nodes.items():
            sorted_lats_pair = sorted(lats.items(), key=lambda item: item[1])
            sorted_pub_lat = [lat for i, lat in sorted_lats_pair]
            num_pubs = float(len(sorted_pub_lat))
            assert(float(x) >= 0 and float(x) <= 100)
            if len(sorted_pub_lat) >= 10:
                lat_x = sorted_pub_lat[int(round(num_pubs*float(x)/100.0)) - 1]
            else:
                lat_x = sorted_pub_lat[int(num_pubs*float(x)/100.0)]
            latencies.append((i, lat_x))

        max_y = max(max_y, max([lat for i, lat in latencies]))
        min_y = min(min_y, min([lat for i, lat in latencies]))
        epochs_lats[e] = latencies

    fig, axs = plt.subplots(ncols=1, nrows=1, constrained_layout=False, figsize=(20,10))
    axs = [axs]
    epochs_time = [i for i in range(len(epochs))]
    title = ""
    plot_figure(epochs_lats, axs[0], epochs_time, min_y, max_y, num_node-1, title, stars)
    # print(epochs_lats)
    fig.savefig('out')



num_node = int(sys.argv[1])
num_epoch = int(sys.argv[2])
prefix = sys.argv[3]

epochs = {}
for e in range(num_epoch):
    nodes = {}
    for i in range(num_node):
        time_path = os.path.join(prefix, "node"+str(i), "epoch"+str(e)+"_time.json")
        msg_path = os.path.join(prefix, "node"+str(i), "epoch"+str(e)+"_msg.json")
        nodes[i] = parse_node_epoch(msg_path, time_path)
    epochs[e] = nodes

plot_time(epochs, num_node, [], 90)
# for e in range(num_epoch):
    # print('Epoch', e, ":" , epochs[e])
    # lat_x = get_diff_Xcent_pubs(node_i, node_lat, x, pubs, proc_delay, ld)
               


# from plot_utility import plot_figure
# from plot_utility import plot_stars_figure
# from plot_utility import get_Xcent_node
# from plot_utility import get_Xcent_node
# from plot_utility import parse_topo
# from plot_utility import parse_adapt
# from plot_utility import parse_file


# if len(sys.argv) < 7:
    # print('Require epoch_dir<str> topo_path<str> x_percent<int(0-100)/avg> unit<node/pub/hash> snapshots_dir<snapshots/snapshots-exploit> epochs<list of int>')
    # sys.exit(0)

# out_dir = sys.argv[1]
# topo = sys.argv[2]
# x_percent = sys.argv[3]
# percent_unit = sys.argv[4] # node or hash
# snapshots_dir = sys.argv[5]
# epochs = [int(i) for i in sys.argv[6:]]

# epoch_dir = os.path.join(out_dir, snapshots_dir)
# adapts = parse_adapt(os.path.join(out_dir, 'adapts'))
# epoch_lats = {}
# max_y = 0
# min_y = 1e8
# num_node = 0
# for e in epochs:
    # epoch_file = os.path.join(epoch_dir, 'epoch'+str(e)+'.txt')
    # lats = parse_file(epoch_file, x_percent, topo, percent_unit)
    # epoch_lats[e] = lats
    # max_y = max(max_y, max([lat for i, lat in lats]))
    # min_y = min(min_y, min([lat for i, lat in lats]))
    # num_node = len(lats)

# fig, axs = plt.subplots(ncols=2, nrows=1, constrained_layout=False, figsize=(20,10))

# exp_name = str(os.path.basename(out_dir))
# context_name = str(os.path.dirname(out_dir))

# title = snapshots_dir + ', ' + str(x_percent)+', '+percent_unit+', '+context_name+', '+str(exp_name)

# patches, _ = plot_figure(epoch_lats, axs[0], epochs, min_y, max_y, num_node-1, title, adapts)

# num_patch_per_row = 10
# interval = int(math.ceil( len(epochs) / num_patch_per_row))
# axs[0].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))

# patches, _ = plot_stars_figure(epoch_lats, axs[1], epochs, min_y, max_y, len(adapts)-1, title, adapts)
# axs[1].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))



# figname = exp_name+"-lat"+str(x_percent)+"-"+percent_unit
# figpath = os.path.join(out_dir, figname)
# lastest_path = os.path.join(out_dir, "latest")

# fig.savefig(figpath)
# fig.savefig(lastest_path)
