import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from plot_utility import plot_lats_cdf
from plot_utility import parse_node_lats
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from plot_utility import parse_adapt


if len(sys.argv) < 3:
    print('need out_dir, epochs')
    sys.exit(1)

out_dir = sys.argv[1]
epochs = [int(e) for e in sys.argv[2:]]
tick_spacing = 50
cdf_spacing = 0.1
num_patch_per_row = 10

exp_name = os.path.basename(out_dir)

snapshots_dir = os.path.join(out_dir, 'snapshots-exploit')
topo_json = os.path.join(out_dir, 'topo.json')
stars = parse_adapt(os.path.join(out_dir, 'adapts'))

cdf_file = os.path.join(out_dir, 'cdfs')

fig, axs = plt.subplots(ncols=2, nrows=len(stars), constrained_layout=False, figsize=(10*2*len(stars),10))
axs = axs.flatten()

# colors
colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0, 0.9, len(epochs))]
patches = []
for i in range(len(epochs)):
    p =  mpatches.Patch(color=colors[i], label=str(epochs[i]))
    patches.append(p) 

for i in range(len(axs)):
    axs[i].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    axs[i].yaxis.set_major_locator(ticker.MultipleLocator(cdf_spacing))
    axs[i].set_ylim(0,1)
    axs[i].set_prop_cycle('color', colors)
    axs[i].set_xlabel('lat')
    axs[i].set_ylabel('prob')
    
    interval = int(math.ceil( len(epochs) / num_patch_per_row))
    axs[i].legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/interval))


max_x = 0
for e in epochs:
    epoch_dist_file = os.path.join(snapshots_dir, 'epoch'+str(e)+'.txt')
    lats, diff_lats = parse_node_lats(stars, epoch_dist_file, topo_json, 'pub')
    for a, star in enumerate(stars):
        max_x = max(max_x, max([lat for i, lat in lats[star]]))
        plot_lats_cdf(lats[star], axs[a*2], 'node' + str(star) + ' cdf ' + exp_name)
        plot_lats_cdf(diff_lats[star], axs[a*2+1], 'node' + str(star) + ' diff lats cdf ' + exp_name)

for a in range(len(axs)):
    axs[a].set_xlim(0, max_x)

plt.savefig(cdf_file)

