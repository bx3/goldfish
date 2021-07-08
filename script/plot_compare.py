#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import matplotlib.patches as mpatches
import math
import matplotlib.ticker as ticker

if len(sys.argv) < 5:
    print('Require x_percent<int(0-100)>, unit<node/hash>, fig-name<str>, exp1 exp2... ')
    sys.exit(0)

# assume epochs are sorted
def plot_figure(percent_X_lats, ax, epochs, xlim, title):
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
    ax.set_xlim(0, xlim)   
    ax.set_title(title, fontsize='small')
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
def parse_file(filename, x, percent_unit):
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

def get_epochs(dirname):
    epochs = []
    for f in os.listdir(dirname):
        e = int(f[5:-4])
        epochs.append(e)
    return epochs

def get_plot_layout(num_exp):
    if num_exp <= 1:
        num_row = 1
        num_col = 1
    elif num_exp <= 2:
        num_row = 1
        num_col = 2
    elif num_exp <= 4:
        num_row = 2
        num_col = 2
    elif num_exp <=6:
        num_row = 2
        num_col = 3
    elif num_exp <=8:
        num_row = 2
        num_col = 4
    else:
        print('Warn. More room to plot')
        sys.exit(0)
    return num_row, num_col



x_percent = int(sys.argv[1])
assert(x_percent >= 0 and x_percent <= 100)
percent_unit = sys.argv[2] # node or hash
fig_name = sys.argv[3]
out_dirs = sys.argv[4:]
num_exp = len(out_dirs)
if num_exp < 2:
    print('need more than 1 exp to compare')
    sys.exit(1)


max_y = 0
num_node = 0

num_row, num_col = get_plot_layout(len(out_dirs))
fig, axs = plt.subplots(ncols=num_col, nrows=num_row, constrained_layout=False, figsize=(num_exp*9,12))

fig_index = 0
for out_dir in out_dirs:

    epoch_dir = os.path.join(out_dir, 'dists')
    epoch_lats = {}
    epochs = get_epochs(epoch_dir)
    epochs = sorted(epochs)
    for e in epochs:
        epoch_file = os.path.join(epoch_dir, 'epoch'+str(e)+'.txt')
        lats = parse_file(epoch_file, x_percent, percent_unit)
        epoch_lats[e] = lats
        max_y = max(max_y, max(lats))
        num_node = len(lats)

    title = str(os.path.basename(out_dir))
    patches = plot_figure(epoch_lats, axs[fig_index], epochs, num_node, title)

    num_patch_per_row = 10
    interval = int(math.ceil( len(epochs) / num_patch_per_row))
    axs[fig_index].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))

    fig_index += 1

fig_index = 0
for out_dir in out_dirs:
    axs[fig_index].set_ylim(0, max_y)
    fig_index += 1

fig.savefig(fig_name)
