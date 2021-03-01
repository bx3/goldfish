#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from collections import namedtuple
import math
import sys
import os
import numpy as np

if len(sys.argv) <=1:
    print('need result dir')
    sys.exit(1)

result_dir = sys.argv[1]
results = os.listdir(result_dir)

#Sample = namedtuple('Sample', ['type','noise','mask','perf']

def plot_data(samples, mask_v, type_v, ax, color):
    noise_ = []
    mask_ = []
    perf_ = []
    for sample in samples:
        data_type, noise, mask, perf = sample 
        if data_type != type_v:
            continue

        if mask == mask_v:
            noise_.append(noise+0.3*random.random())
            mask_.append(mask)
            perf_.append(perf)
    sorted_zipped = sorted(zip(noise_, perf_))
    sorted_noise_ = [a for a,_ in sorted_zipped]
    sorted_perf_ = [b for _,b in sorted_zipped]
    ax.plot(sorted_noise_, sorted_perf_, color=color)   #s=12, 

def read_perf(name):
    W_perfs = []
    H_perfs = []
    with open(name) as f:
        i = 0
        for line in f:
            if i == 0:
                W_perfs = [float(a) for a in line.split()]
                i += 1
            else:
                H_perfs = [float(a) for a in line.split()]
    return W_perfs, H_perfs

samples = []
perf = []
exp_types = []
for result_file in results:
    num_mask = -1
    noise_std = -1
    exp_type = None
    note = result_file.split('-')[-1]
    for token in result_file.split('-'):
        if 'noise' in token:
            noise_std = int(token[5:])
        elif 'mask' in token:
            num_mask = int(token[:-4])
        elif 'linear' in token or 'unif' in token or 'log-unif' in token:
            exp_type = token
    if exp_type + note not in exp_types:
        exp_types.append(exp_type + note)
    sample_tag = exp_type + note
    print(result_file, noise_std, num_mask, exp_type)
    W_perfs, H_perfs = read_perf(result_dir +'/'+ result_file)
    last_five_avg = sum(W_perfs[:-5])/len(W_perfs[:-5])
    perf.append((sample_tag, noise_std, num_mask, last_five_avg))
samples += perf

print(exp_types)
print(perf)
# bandit lcb
# lcb_perf  = [('lcb',0,0,1), ('lcb',0,5,1), ('lcb',0,10,1), ('lcb',0,15,1)]
# lcb_perf += [('lcb',5,0,1), ('lcb',5,5,1), ('lcb',5,10,1), ('lcb',5,15,0.78)]
# lcb_perf += [('lcb',10,0,1), ('lcb',10,5,0.8), ('lcb',10,10,0.86), ('lcb',10,15,0.5)]
# lcb_perf += [('lcb',15,0,0.65), ('lcb',15,5,0.7)]
# samples += lcb_perf

fig, axs = plt.subplots(2,2)
# colors = ['red', 'blue', 'green']
colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0, 0.9, len(exp_types))]

# plot_data(samples, 0, 'rand', axs[0,0], 'blue')
# plot_data(samples, 0, 'mean', axs[0,0], 'green')
for i in range(len(exp_types)):
    plot_data(samples, 0, exp_types[i],  axs[0,0], colors[i])

axs[0,0].set_title('num mask per row = 0', fontsize='small')
axs[0,0].set_ylabel('perf', fontsize='small')

# plot_data(samples, 5, 'rand', axs[0,1], 'blue')
# plot_data(samples, 5, 'mean', axs[0,1], 'green')
for i in range(len(exp_types)):
    plot_data(samples, 5, exp_types[i],  axs[0,1], colors[i])
    axs[0,1].set_title('num mask per row= 5', fontsize='small')

# plot_data(samples, 10, 'rand', axs[1,0], 'blue')
# plot_data(samples, 10, 'mean', axs[1,0], 'green')
for i in range(len(exp_types)):
    plot_data(samples, 10, exp_types[i],  axs[1,0], colors[i])
axs[1,0].set_title('num mask per row= 10', fontsize='small')
axs[1,0].set_ylabel('perf', fontsize='small')
axs[1,0].set_xlabel('noise std', fontsize='small')


# plot_data(samples, 15, 'rand', axs[1,1], 'blue')
# plot_data(samples, 15, 'mean', axs[1,1], 'green')
for i in range(len(exp_types)):
    plot_data(samples, 15, exp_types[i],  axs[1,1], colors[i])
axs[1,1].set_title('num mask per row= 15', fontsize='small')
axs[1,1].set_xlabel('noise std', fontsize='small')

# # ax.set(ylabel='perf at iter 50', xlabel='noise std')
# # ax.set_title('perf, noise. label is num mask')

# rand_patch = mpatches.Patch(color='blue', label='rand')
patches = []
for i in range(len(exp_types)):
    patch = mpatches.Patch(color=colors[i], label=exp_types[i])
    patches += [patch]

# handles, labels = ax[3].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center')

fig.legend(loc='lower center', handles=patches, fontsize='small', ncol=len(patches)) #, ncol= math.ceil(len(patches)/2)
plt.tight_layout()
plt.show()


