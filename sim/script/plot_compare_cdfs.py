import sys
import os
import numpy as np
from plot_utility import plot_cdfs
import matplotlib.pyplot as plt
from plot_utility import parse_adapt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

if len(sys.argv) < 5:
    print('Require fig-name<str>, exp1, exp2, epochs... ')
    sys.exit(0)

figname = sys.argv[1]
out_dirs = sys.argv[2:4]
epochs = [int(e) for e in sys.argv[4:]]
num_compare_star = 10

stars = parse_adapt(os.path.join(out_dirs[0], 'adapts'))   
stars_2 = parse_adapt(os.path.join(out_dirs[1], 'adapts'))   
assert(stars == stars_2)

if len(stars) > num_compare_star:
    stars = np.random.choice(stars, num_compare_star, replace=False)

fig, axs = plt.subplots(ncols=2, nrows=len(stars), constrained_layout=False, figsize=(10*2,10*len(stars)))
axs = axs.flatten()
styles = ['-', '--']
line_patches = []
for i,out_dir in enumerate(out_dirs):
    plot_cdfs(out_dir, stars, epochs, axs, styles[i])
    title = str(os.path.dirname(out_dir))
    p = Line2D([0], [0], color='k', linestyle=styles[i], label=title)
    line_patches.append(p) 
axs[0].legend(handles=line_patches, title='context', bbox_to_anchor=(1, 1.2), loc='upper left', fontsize='xx-small')
plt.savefig(figname)


