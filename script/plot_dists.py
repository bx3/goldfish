#!/usr/bin/env python
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from collections import defaultdict
import matplotlib.patches as mpatches
import math
from plot_topo import plot_topology
import numpy as np

if len(sys.argv) < 3:
    print('require input_file, input json only works for fixed miners')
    sys.exit(1)

def plot_dist(star_i, dists_hist, ax, record_epochs):
    num_epoch = len(dists_hist)

    # print(dists_hist)
    pubs_dist = defaultdict(list)
    x_peer = []

    pubs = set()
    for e, dists in dists_hist:
        pubs = pubs.union(dists.keys())
    pubs = sorted(list(pubs))
    pubs_line_dist = {}
    # colors
    # colormap = plt.cm.nipy_spectral
    # colors = [colormap(i) for i in np.linspace(0, 0.9, len(pubs))]
    # patches = []
    # for i in range(len(pubs)):
        # p =  mpatches.Patch(color=colors[i], label=str(pubs[i]))
        # patches.append(p) 

    x_peers = []
    for e, dists in dists_hist:
        if e not in record_epochs:
            continue
        peers = []
        for m in pubs:
            if m in dists:
                peer, topo_length, line_len = dists[m]
                diff = topo_length - line_len
                pubs_dist[m].append(diff)
                peers.append(str(peer))
                if m not in pubs_line_dist:
                    pubs_line_dist[m] = line_len
                else:
                    assert(pubs_line_dist[m] == line_len)
            else:
                # pub m  
                print('pub', m, 'is not included in the pickle')
                pubs_dist[m].append(10000)
        # x_peers.append(str(e) + '  |' + ','.join(peers))
        x_peers.append(str(e))


    pubs_line_dist = [(m,round(d)) for m,d in sorted(pubs_line_dist.items())]
    # print(pubs)
    # print(x_peers)
    # print(pubs_dist) 
    index = pd.Index(x_peers, name='epoch, peers')
    
    df = pd.DataFrame(pubs_dist, index=index)
    if len(pubs) <= 10:
        df.plot(ax=ax, kind='bar', stacked=True)
    else:
        df.plot(ax=ax, kind='bar', stacked=True, legend=False)
    for index, label in enumerate(ax.get_xticklabels()):
        if index % 1 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)
    label.set_visible(True)

    ax.set_ylabel('dist')
    basename = os.path.basename(filename)
    ax.set_title(basename + ' node'+str(star_i))
    # if len(pubs) <= 10:
        # ax.legend(pubs_line_dist, title='pubs (id,dist)', loc='upper right') #bbox_to_anchor=(1.0,1), 
    
    # plt.tight_layout()


# script starts
filename = sys.argv[1]
topo_json = sys.argv[2]
record_epochs = [int(e) for e in sys.argv[3:]]

stars_dists_hist = None
max_plot = 10 

with open(filename, 'rb') as f:
    stars_dists_hist = pickle.load(f)

sorted_stars = sorted(stars_dists_hist.keys())
num_plot = min(len(sorted_stars), max_plot)
sorted_stars = list(np.random.choice(sorted_stars, num_plot, replace=False))

num_plot = len(sorted_stars)  + 1

num_row = min(3, num_plot)
num_col = int(math.ceil(num_plot/float(num_row)))

fig, axs = plt.subplots(ncols=num_col, nrows=num_row, figsize=(10*num_col,10*num_row))

if num_row == 1 and num_col == 1:
    axs = [axs]
else:
    axs = axs.flatten()

ax_i = 0
plot_topology(topo_json, axs[0], sorted_stars)

for i in range(len(sorted_stars)):
    if i+1 == len(axs):
        break
    ax = axs[i+1]
    star_i = sorted_stars[i]
    dists_hist = stars_dists_hist[star_i]
    plot_dist(star_i, dists_hist, ax, record_epochs)

plt.savefig(filename)

