#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import os
import json
import numpy as np
import matplotlib.patches as mpatches
import math
import matplotlib.ticker as ticker

from plot_utility import plot_figure
from plot_utility import plot_stars_figure
from plot_utility import get_Xcent_node
from plot_utility import get_Xcent_node
from plot_utility import parse_topo
from plot_utility import parse_adapt
from plot_utility import parse_file


if len(sys.argv) < 7:
    print('Require epoch_dir<str> topo_path<str> x_percent<int(0-100)/avg> unit<node/pub/hash> snapshots_dir<snapshots/snapshots-exploit> epochs<list of int>')
    sys.exit(0)

out_dir = sys.argv[1]
topo = sys.argv[2]
x_percent = sys.argv[3]
percent_unit = sys.argv[4] # node or hash
snapshots_dir = sys.argv[5]
epochs = [int(i) for i in sys.argv[6:]]

epoch_dir = os.path.join(out_dir, snapshots_dir)
adapts = parse_adapt(os.path.join(out_dir, 'adapts'))
epoch_lats = {}
max_y = 0
min_y = 1e8
num_node = 0
for e in epochs:
    epoch_file = os.path.join(epoch_dir, 'epoch'+str(e)+'.txt')
    lats = parse_file(epoch_file, x_percent, topo, percent_unit)
    epoch_lats[e] = lats
    max_y = max(max_y, max([lat for i, lat in lats]))
    min_y = min(min_y, min([lat for i, lat in lats]))
    num_node = len(lats)

fig, axs = plt.subplots(ncols=2, nrows=1, constrained_layout=False, figsize=(20,10))

exp_name = str(os.path.basename(out_dir))
context_name = str(os.path.dirname(out_dir))

title = snapshots_dir + ', ' + str(x_percent)+', '+percent_unit+', '+context_name+', '+str(exp_name)

patches, _ = plot_figure(epoch_lats, axs[0], epochs, min_y, max_y, num_node-1, title, adapts)

num_patch_per_row = 10
interval = int(math.ceil( len(epochs) / num_patch_per_row))
axs[0].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))

patches, _ = plot_stars_figure(epoch_lats, axs[1], epochs, min_y, max_y, len(adapts)-1, title, adapts)
axs[1].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))



figname = exp_name+"-lat"+str(x_percent)+"-"+percent_unit
figpath = os.path.join(out_dir, figname)
lastest_path = os.path.join(out_dir, "latest")

fig.savefig(figpath)
fig.savefig(lastest_path)
