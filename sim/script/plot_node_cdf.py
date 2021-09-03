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
from plot_utility import plot_cdfs


if len(sys.argv) < 3:
    print('need out_dir, epochs')
    sys.exit(1)

out_dir = sys.argv[1]
epochs = [int(e) for e in sys.argv[2:]]

stars = parse_adapt(os.path.join(out_dir, 'adapts'))
num_plotting_stars = 10
if len(stars) >  num_plotting_stars:
    stars = sorted(np.random.choice(stars, num_plotting_stars, replace=False))

fig, axs = plt.subplots(ncols=2, nrows=len(stars), constrained_layout=False, figsize=(10*2,10*len(stars)))
axs = axs.flatten()

plot_cdfs(out_dir, stars, epochs, axs, '-')
cdf_file = os.path.join(out_dir, 'cdfs')
plt.savefig(cdf_file)

