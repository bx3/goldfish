#!/usr/bin/env python
import matplotlib.pyplot as plt
import sys
import os
import json
import pickle
import math
from collections import defaultdict
from plot_utility import get_topo_loc_delay
from plot_utility import parse_file 
from plot_utility import parse_adapt
import statistics

if len(sys.argv) < 7:
    print('Require args')
    print('rand-100node-3pub-20_0proc')
    sys.exit(1)

start_seed = int(sys.argv[1])
num_seed = int(sys.argv[2])
num_star = sys.argv[3]
context_dir = sys.argv[4]
topo_prefix = sys.argv[5]
x_percent = 90
percent_unit = 'hash'

interested_epochs = [int(e) for e in sys.argv[6:]]

epoch_data = {}

for e in interested_epochs:
    percent_25 = []
    percent_50 = []
    percent_75 = []
    mean_list = []

    for s in range(start_seed, start_seed + num_seed):
        # filename = topo_prefix + '-'+str(s) + 'seed.json'
        star_i = None
        star_exploits = None

        seed_dir =  topo_prefix + '-'+str(s) + 'seed-'+num_star+ 'stars' 
        adapt_file = os.path.join(context_dir, seed_dir, 'adapts')
        stars = parse_adapt(adapt_file)

        if not os.path.exists(os.path.join(context_dir, seed_dir, 'latest.png')):
            print('seed', s, 'not finish run')
            continue

        topo_json = os.path.join(context_dir, seed_dir, 'topo.json')
        loc, proc_delay, ld, pub_prob = get_topo_loc_delay(topo_json)
        pubs = set()
        for p, prob in pub_prob.items():
            if prob > 0:
                pubs.add(p)

        epoch_lats = {}
        epoch_dir = os.path.join(context_dir, seed_dir, 'snapshots-exploit')
    
        epoch_file = os.path.join(epoch_dir, 'epoch'+str(e)+'.txt')
        lats = parse_file(epoch_file, x_percent, topo_json, percent_unit)

        sorted_node_lats = sorted(lats, key=lambda item: item[1])

        sorted_lats = [lat for i, lat in sorted_node_lats if i in stars]
        sorted_node = [i for i, lat in sorted_node_lats if i in stars]

        assert(len(sorted_lats) == len(stars))
        print('num star', len(stars))

        index_25 = int(math.ceil(len(sorted_lats) / 4.0)-1)
        index_50 = int(math.ceil(len(sorted_lats) / 2.0)-1)
        index_75= int(math.ceil(len(sorted_lats) / 4.0*3.0)-1)


        percent_25.append(sorted_lats[index_25])
        percent_50.append(sorted_lats[index_50])
        percent_75.append(sorted_lats[index_75])
        mean_list.append(sum(sorted_lats)/len(sorted_lats))

    p25 = sum(percent_25) / len(percent_25)
    p50 = sum(percent_50) / len(percent_50)
    p75 = sum(percent_75) / len(percent_75)
    p_mean = sum(mean_list) / len(mean_list)

    m25 = statistics.median(percent_25)
    m50 = statistics.median(percent_50)
    m75 = statistics.median(percent_75)
    m_mean = statistics.median(mean_list)

    print('data size', len(mean_list))
    output = 'epoch:'+ str(e)+" "+ "mean percentile 25: " + str(round(p25)) + ".  median percentile 25: " + str(round(m25)) + '\n'
    print(output)
    output = 'epoch:'+ str(e)+" "+ "mean percentile 50: " + str(round(p50)) + ".  median percentile 50: " + str(round(m50)) + '\n'
    print(output)
    output = 'epoch:'+ str(e)+" "+ "mean percentile 75: " + str(round(p75)) +    ".  median percentile 75: " + str(round(m75)) + '\n'
    print(output)
    output = 'epoch:'+ str(e)+" "+ "mean mean         : " + str(round(p_mean)) + ".  median mean         : " + str(round(m_mean)) + '\n'
    print(output)





         



    

