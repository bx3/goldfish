#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import matplotlib.patches as mpatches
import math
import matplotlib.ticker as ticker

if len(sys.argv) < 1:
    print('Require data directory')
    sys.exit(0)

method = 'subset'

def plot_figure(dirname, method, ax, snapshot_point, title, ylim, xlim):
    subset_data ={}
    ax.set_prop_cycle('color', colors)
    for r in snapshot_point:
        filename = dirname + "/result90unhash_" + method + "V1Round" + str(r) + ".txt"
        buff = []
        # print(filename)
        f = open(filename,'r',errors='replace')
        line=f.readlines()
        a=line[0].strip().split("  ")
        for j in range(len(a)):
            buff.append(int(float(a[j])))
        subset_data[r]=sorted(buff)
        f.close()

    for i, d in subset_data.items():
        ax.grid(True)
        ax.plot(d) #, label="round"+str(i)
    tick_spacing = 50
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    name = title.split('/')[-1]
    ax.set_title(name)
    ax.legend()

def get_y_lim(dirname, method, snapshot_point, min_y, max_y):
    num_node = 0
    for r in snapshot_point:
        filename = dirname + "/result90unhash_" + method + "V1Round" + str(r) + ".txt"
        with open(filename,'r') as f:
            line=f.readlines()
            a=line[0].strip().split("  ")
            num_node = len(a)
            for j in range(len(a)):
                n = int(float(a[j]))
                if max_y == None or n > max_y:
                    max_y = n
                if min_y == None or n < min_y:
                    min_y = n
    return min_y, max_y, num_node


datadir_list = []
for i in range(1, len(sys.argv)):
    datadir_list.append(sys.argv[i])

snapshot_point = [0,8,16,32,64,96]

num_row = 1
num_col = len(datadir_list)
min_y = None 
max_y = None
for dirname in datadir_list:
    dirpath = dirname
    min_y, max_y, num_node = get_y_lim(dirpath, method, snapshot_point, min_y, max_y)

ylim = [200,650]#[min_y, max_y]
xlim = [0, num_node]
# print(ylim)
# print(num_row, num_col)

data_dirname = datadir_list
num_row = 2
num_col = 3
num_exp = len(data_dirname)
print('num_exp', num_exp)
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

fig, axs = plt.subplots(ncols=num_col, nrows=num_row, constrained_layout=False, figsize=(18,9))
colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0, 0.9, len(snapshot_point))]
patches = []

for i in range(len(snapshot_point)):
    p =  mpatches.Patch(color=colors[i], label=str(snapshot_point[i]))
    patches.append(p) 

max_patch = mpatches.Patch(color='red', label='max')
min_patch = mpatches.Patch(color='green', label='min')
mean_patch = mpatches.Patch(color='blue', label='mean')

# print('h',len(axs), num_exp)
i = 0
c = 0
r = 0
for dirname in data_dirname:
    c = int(i / num_col)
    r = i % num_col
    #print(c, )
    dirpath = dirname
    
    if num_row ==1 and num_col == 1:
        plot_figure(dirpath, method, axs, snapshot_point, dirname, ylim, xlim)
    elif len(axs) == num_exp:
        plot_figure(dirpath, method, axs[i], snapshot_point, dirname, ylim, xlim)
    else:
        plot_figure(dirpath, method, axs[c, r], snapshot_point, dirname, ylim, xlim)
    i += 1
    if i >= num_row* num_col:
        break


fig.legend(loc='lower center', handles=patches, fontsize='small', ncol= math.ceil(len(patches)/2))
plt.show()
#plt.savefig("subset64_2.png")
print("finish")
