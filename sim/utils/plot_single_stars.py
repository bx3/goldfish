#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import os
import json
import numpy as np
import matplotlib.patches as mpatches
import math
import matplotlib.ticker as ticker

if len(sys.argv) < 6:
    print('Require epoch_dir<str> topo_path<str> x_percent<int(0-100)> unit<node/pub/hash> epochs<list of int>')
    sys.exit(0)

# assume epochs are sorted
def plot_stars_figure(percent_X_lats, ax, epochs, min_y, max_y, xlim, title, stars, pubs):
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(epochs))]
    patches = []

    for i in range(len(epochs)):
        p =  mpatches.Patch(color=colors[i], label=str(epochs[i]))
        patches.append(p) 

    tick_spacing = 50
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_prop_cycle('color', colors)

    for e in epochs:
        lats = percent_X_lats[e]
        sorted_node_lats = sorted(lats, key=lambda item: item[1])
        sorted_lats = [lat for i, lat in sorted_node_lats if i in pubs]
      
        ax.plot(sorted_lats)

    ax.grid(True)
    ax.set_ylim(min_y, max_y)
    ax.set_xlim(0, xlim)
    ax.set_title(title, fontsize='small')
    ax.legend()
    return patches

def get_num_node(dirname, method, snapshot_point):
    num_node = None 
    for r in snapshot_point:
        filename = dirname + "/result90unhash_" + method + "V1Round" + str(r) + ".txt"
        with open(filename,'r') as f:
            line=f.readlines()
            a=line[0].strip().split("  ")
            if num_node == None:
                num_node = len(a)
            else:
                assert(num_node == len(a))
    return num_node

# x is float
def get_Xcent_node(lats, x):
    sorted_lats_pair = sorted(lats.items(), key=lambda item: item[1])
    sorted_lat = [lat for i, lat in sorted_lats_pair]
    if len(sorted_lat) >= 10:
        lat_x = sorted_lat[int(round(len(sorted_lat)*float(x)/100.0)) - 1]
    else:
        lat_x = sorted_lat[int(len(sorted_lat)*float(x)/100.0)]
    return lat_x

def get_Xcent_pubs(lats, x, pubs):
    sorted_lats_pair = sorted(lats.items(), key=lambda item: item[1])
    sorted_pub_lat = [lat for i, lat in sorted_lats_pair if i in pubs]
    num_pubs = float(len(sorted_pub_lat))

    if len(sorted_pub_lat) >= 10:
        lat_x = sorted_pub_lat[int(round(num_pubs*float(x)/100.0)) - 1]
    else:
        lat_x = sorted_pub_lat[int(num_pubs*float(x)/100.0)]
    return lat_x


def parse_topo(topo_json):
    num_pub = 0
    num_node = None
    pubs = []
    with open(topo_json) as config:
        data = json.load(config)
        nodes = data['nodes']
        summary = data['summary']
        num_node = summary['num_node']
        role = {}

        for node in nodes:
            if node["role"] == 'PUB':
                num_pub += 1
                pubs.append(node['id'])
    return num_pub, pubs

def parse_adapt(filename):
    adapts = []
    with open(filename) as f:
        for line in f:
            adapt = int(line.split()[0])
            adapts.append(adapt)
    return adapts

# return a list whose i-th entry represnts latency to reach 90cent nodes for node i
def parse_file(filename, x, topo, percent_unit):
    latency = []
    num_pub, pubs = parse_topo(topo)
    with open(filename) as f:
        node_i = 0
        for line in f:
            tokens = line.split()   
            node_lat = {}
            for i in range(len(tokens)):
                node_lat[i] = float(tokens[i])
            if percent_unit == 'node':
                lat_x = get_Xcent_node(node_lat, x)
            elif percent_unit == 'pub':
                lat_x = get_Xcent_pubs(node_lat, x, pubs)
            elif percent_unit == 'hash':
                print('Not implemented. topo json file needs hash')
                sys.exit(1)
            else:
                print('Unknown percent unit', percent_unit)
                sys.exit(1)
            latency.append((node_i, lat_x))
            node_i += 1
    return latency 


out_dir = sys.argv[1]
topo = sys.argv[2]
x_percent = int(sys.argv[3])
assert(x_percent >= 0 and x_percent <= 100)
percent_unit = sys.argv[4] # node or hash
epochs = [int(i) for i in sys.argv[5:]]

epoch_dir = os.path.join(out_dir, 'snapshots')
adapts = parse_adapt(os.path.join(out_dir, 'adapts'))
epoch_lats = {}
max_y = 0
min_y = 1e8
num_node = 0

num_pub, pubs = parse_topo(topo)
for e in epochs:
    epoch_file = os.path.join(epoch_dir, 'epoch'+str(e)+'.txt')
    lats = parse_file(epoch_file, x_percent, topo, percent_unit)
    epoch_lats[e] = lats
    max_y = max(max_y, max([lat for i, lat in lats if i in pubs]))
    min_y = min(min_y, min([lat for i, lat in lats if i in pubs]))
    num_node = len(lats)

fig, axs = plt.subplots(ncols=1, nrows=1, constrained_layout=False, figsize=(10,10))

title = str(os.path.basename(out_dir))


patches = plot_stars_figure(epoch_lats, axs, epochs, min_y, max_y, num_pub, title, adapts, pubs)

num_patch_per_row = 10
interval = int(math.ceil( len(epochs) / num_patch_per_row))
axs.legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))

figname = title+"-lat"+str(x_percent)+"-"+percent_unit
figpath = os.path.join(out_dir, figname)
lastest_path = os.path.join(out_dir, "latest")

fig.savefig(figpath)
fig.savefig(lastest_path)
