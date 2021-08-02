#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import matplotlib.patches as mpatches
import math
import matplotlib.ticker as ticker

from plot_utility import * 


if len(sys.argv) < 9:
    print('Require x_percent<int(0-100)/avg>, unit<node/hash>, fig-name<str>, exp1 exp2, topo<str> snapshots_dir<snapshots/snapshots-exploit> epochs... ')
    sys.exit(0)

x_percent = sys.argv[1]
percent_unit = sys.argv[2] # node or hash
fig_name = sys.argv[3]
out_dirs = sys.argv[4:6]
topo = sys.argv[6]
snapshot_dir = sys.argv[7]

epochs = [int(e) for e in sys.argv[8:]]
num_plot = len(out_dirs)

if num_plot < 2:
    print('need more than 1 exp to compare')
    sys.exit(1)

max_y = 0
num_node = 0

num_row = 2 # min(2, num_plot)
num_col = 3 # int(math.ceil(num_plot/float(num_row)))

fig, axs = plt.subplots(ncols=num_col, nrows=num_row, figsize=(10*num_col,10*num_row))

axs = axs.flatten()

fig_index = 0

max_y = 0
min_y = 1e8
num_node = 0
exp_data = {}
num_patch_per_row = 8 
for out_dir in out_dirs:
    epoch_dir = os.path.join(out_dir, snapshot_dir)
    adapts = parse_adapt(os.path.join(out_dir, 'adapts'))

    epoch_lats = {}
    for e in epochs:
        epoch_file = os.path.join(epoch_dir, 'epoch'+str(e)+'.txt')
        lats = parse_file(epoch_file, x_percent, topo, percent_unit)
        epoch_lats[e] = lats
        max_y = max(max_y, max([lat for i, lat in lats]))
        min_y = min(min_y, min([lat for i, lat in lats]))
        num_node = len(lats)
    exp_data[out_dir] = epoch_lats

all_sorted_lats = {}
num_pub, pubs = parse_topo(topo)
# plot all nodes
for out_dir in out_dirs:
    adapts = parse_adapt(os.path.join(out_dir, 'adapts'))
    epoch_lats = exp_data[out_dir]
    title = str(os.path.dirname(out_dir))+' '+str(os.path.basename(out_dir))
    patches, sorted_lats = plot_figure(epoch_lats, axs[fig_index], epochs, min_y, max_y, num_node-1, title, adapts)
    all_sorted_lats[out_dir] = sorted_lats
    interval = int(math.ceil( len(epochs) / num_patch_per_row))
    axs[fig_index].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))
    fig_index += 1

# plot difference
names = '\n'.join(out_dirs)
title = 'diff, ' + str(x_percent)+', '+percent_unit +  '\n' + names
patches = plot_diff_lats(all_sorted_lats, out_dirs, axs[fig_index], epochs, num_node-1, title)
interval = int(math.ceil( len(epochs) / num_patch_per_row))
axs[fig_index].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))
fig_index += 1

# plot stars only
all_sorted_stars_lats = {}
for out_dir in out_dirs:
    adapts = parse_adapt(os.path.join(out_dir, 'adapts'))
    epoch_lats = exp_data[out_dir]
    title = 'stars ' + str(os.path.dirname(out_dir))+' '+str(os.path.basename(out_dir))
    patches, sorted_lats = plot_stars_figure(epoch_lats, axs[fig_index], epochs, min_y, max_y, len(adapts)-1, title, adapts)
    all_sorted_stars_lats[out_dir] = sorted_lats
    interval = int(math.ceil( len(epochs) / num_patch_per_row))
    axs[fig_index].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))
    fig_index += 1



title = 'diff, ' + str(x_percent)+', '+percent_unit +  '\n ' + names 
patches = plot_diff_lats(all_sorted_stars_lats, out_dirs, axs[fig_index], epochs, len(adapts)-1, title)
interval = int(math.ceil( len(epochs) / num_patch_per_row))
axs[fig_index].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))
fig_index += 1


fig.savefig(fig_name)
