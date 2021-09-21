#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
import os
import json
import pickle
from collections import defaultdict
from plot_utility import get_topo_loc_delay

if len(sys.argv) < 7:
    print('Require args')
    print('rand-100node-3pub-20_0proc')
    sys.exit(1)

start_seed = int(sys.argv[1])
num_seed = int(sys.argv[2])
num_epoch = int(sys.argv[3])
approx_ratio = float(sys.argv[4]) # below is good
context_dir = sys.argv[5]
topo_prefix = sys.argv[6]


good_epochs = defaultdict(int)
approx_good_epochs = defaultdict(int)

conns = []

for s in range(start_seed, start_seed + num_seed):
    filename = topo_prefix + '-'+str(s) + 'seed.json'
    topo_json = os.path.join('topo', filename)

    loc, proc_delay, ld, pub_prob = get_topo_loc_delay(topo_json)
    pubs = set()
    for p, prob in pub_prob.items():
        if prob > 0:
            pubs.add(p)

    star_i = None
    star_exploits = None

    seed_dir =  topo_prefix + '-'+str(s) + 'seed-1stars' 
    adapt_file = os.path.join(context_dir, seed_dir, 'adapts')
    with open(adapt_file) as f:
        star_i = int(f.readline().split()[0])

    if not os.path.exists(os.path.join(context_dir, seed_dir, 'latest.png')):
        print('seed', s, 'not finish run')
        continue

    graph_dir = os.path.join(context_dir, seed_dir, 'graphs')
    for e in range(num_epoch):
        graph_epoch_json = os.path.join(graph_dir, 'epoch'+str(e)+'.json')
        nodes = None
        with open(graph_epoch_json) as f:
            nodes = json.load(f)
        star = nodes[star_i]
        assert(star['node'] == star_i)
        star_exploits = set(star['exploits'])
        conns.append(star_exploits)
        diff = pubs.difference(star_exploits)

        if len(diff) == 0:
            good_epochs[s] += 1

    # get good approx
    dist_star_file = os.path.join(context_dir, seed_dir, seed_dir)
    with open(dist_star_file, 'rb') as f:
        stars_dists_hist = pickle.load(f)
    x_peers = []

    e, init_dists = stars_dists_hist[star_i][0]
    assert(e==0)
    init_diff_sum =  0
    for m in pubs:
        peer, topo_length, line_len = init_dists[m]
        diff = topo_length - line_len
        init_diff_sum += diff

    for e, dists in stars_dists_hist[star_i]:
        peers = []
        sum_diff = 0
        for m in pubs:
            if m in dists:
                peer, topo_length, line_len = dists[m]
                diff = topo_length - line_len
                sum_diff += diff
            else:
                print('pub', m, 'is not included in the pickle')
                sys.exit(1)

        if sum_diff / float(init_diff_sum) < approx_ratio:
            approx_good_epochs[s] += 1

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12,4))



# strict good
strict_exploration_epoch_counts = [num_epoch-i for i in good_epochs.values()]
approx_exploration_epoch_counts = [num_epoch-i for i in approx_good_epochs.values()]

ylim = 0.028

n, bins, patches = axs[0].hist(strict_exploration_epoch_counts, 100, density=True, facecolor='g', alpha=0.75)
axs[0].set_xlim(0,num_epoch)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('density')
axs[0].set_title('number epochs without optimal connections', fontsize=10)
tick_spacing = 0.02 
axs[0].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# axs[0].set_ylim(0,ylim)


n, bins, patches = axs[1].hist(approx_exploration_epoch_counts, 100, density=True, facecolor='g', alpha=0.75)
axs[1].set_xlim(0,num_epoch)
axs[1].set_xlabel('epochs')
axs[1].set_title('number epochs sufficiently far from optimal', fontsize=10)
axs[1].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# axs[1].set_ylim(0,ylim)



a = [i for i in strict_exploration_epoch_counts if i > 96]
print(a)

print('cdf less than 100 epoch', len(a) / len(strict_exploration_epoch_counts))

a = [i for i in approx_exploration_epoch_counts if i > 48]
print(a)

print('approx cdf less than 100 epoch', len(a) / len(approx_exploration_epoch_counts))


plt.savefig('optimal-single-star')


