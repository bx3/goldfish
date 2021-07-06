#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import matplotlib.patches as mpatches
import math
import matplotlib.ticker as ticker

if len(sys.argv) < 6:
    print('Require epoch_dir<str> topo_path<str> x_percent<int(0-100)>, unit<node/hash> epochs<list of int>')
    sys.exit(0)

# assume epochs are sorted
def plot_figure(percent_X_lats, ax, epochs, ylim, xlim, title):
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
        sorted_lats = sorted(lats)
        ax.plot(sorted_lats)

    ax.grid(True)
    ax.set_ylim(0, ylim)
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

# return a list whose i-th entry represnts latency to reach 90cent nodes for node i
def parse_file(filename, x, topo, percent_unit):
    latency = []
    with open(filename) as f:
        for line in f:
            tokens = line.split()   
            node_lat = {}
            for i in range(len(tokens)):
                node_lat[i] = float(tokens[i])
            if percent_unit == 'node':
                lat_x = get_Xcent_node(node_lat, x)
            elif percent_unit == 'hash':
                print('Not implemented')
                sys.exit(1)
            else:
                print('Unknown percent unit', percent_unit)
                sys.exit(1)
            latency.append(lat_x)
    return latency 


out_dir = sys.argv[1]
topo = sys.argv[2]
x_percent = int(sys.argv[3])
assert(x_percent >= 0 and x_percent <= 100)
percent_unit = sys.argv[4] # node or hash
epochs = [int(i) for i in sys.argv[5:]]

epoch_dir = os.path.join(out_dir, 'dists')
epoch_lats = {}
max_y = 0
num_node = 0
for e in epochs:
    epoch_file = os.path.join(epoch_dir, 'epoch'+str(e)+'.txt')
    lats = parse_file(epoch_file, x_percent, topo, percent_unit)
    epoch_lats[e] = lats
    max_y = max(max_y, max(lats))
    num_node = len(lats)

fig, axs = plt.subplots(ncols=1, nrows=1, constrained_layout=False, figsize=(9,12))

title = str(os.path.basename(out_dir))
patches = plot_figure(epoch_lats, axs, epochs, max_y, num_node, title)

num_patch_per_row = 10
interval = int(math.ceil( len(epochs) / num_patch_per_row))
axs.legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))

figname = title+"-lat"+str(x_percent)+"-"+percent_unit
figpath = os.path.join(out_dir, figname)
lastest_path = os.path.join(out_dir, "latest")

fig.savefig(figpath)
fig.savefig(lastest_path)
