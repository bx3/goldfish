#!/usr/bin/env python
import numpy as np
import sys
import random 

seed_num = int(sys.argv[2])
np.random.seed(seed_num)
random.seed(seed_num)

import time
import config
import initnetwork
import readfiles
import math
import tester

from experiment import Experiment
def run():
    if len(sys.argv) < 4:
        print('Need arguments.')
        print('./testbed subcommand[run/complete_graph] seed[int] num_out[int] num_region[int] output_dir[str] use_node_hash[y/n] rounds[intList]')
        print('./testbed run 1 8 4 n 0 1 2 3 4 5')
        sys.exit()

    subcommand = sys.argv[1]
    data_path = sys.argv[3]
    out_lim = int(sys.argv[4])
    num_region = int(sys.argv[5])
    use_node_hash = sys.argv[6]=='y'

    record_epochs = [int(i) for i in sys.argv[7:]]
    max_epoch = max(record_epochs) +1
    
    window = int(config.window_constant * num_region * math.ceil(math.log(config.num_node))) # T > L log N
    num_msg = 1 # int(window / config.num_batch)


    [ node_delay, 
      node_hash, link_delay, 
      neighbor_set, IncomingLimit, 
      outs_neighbors, in_lims, 
      bandwidth] = initnetwork.GenerateInitialNetwork(
            config.network_type,
            config.num_node, 
            subcommand,
            out_lim)



    if config.use_reduce_link:
        print("\033[91m" + 'Use reduced link latency' + "\033[0m")
        initnetwork.reduce_link_latency(config.num_node, int(0.2*config.num_node), link_delay)
    else:
        print("\033[93m" + 'Not use reduced link latency'+ "\033[0m")

    # print(node_hash)
    # for i in range(len(link_delay)):
        # print(link_delay[i])
    # for i in range(len(outs_neighbors)):
        # print(i, outs_neighbors[i])

    if not use_node_hash:
        print("\033[93m" + 'Not Use asymmetric node hash'+ "\033[0m")
        node_hash = None 
    else:
        print("\033[91m" + 'Use asymmetric node hash'+ "\033[0m")

    if config.use_matrix_completion:
        print("\033[93m" + 'Use matrix completion'+ "\033[0m")
        if config.use_abs_time:
            print("\033[93m" + '\tuse absolute time'+ "\033[0m")
        else:
            print("\033[93m" + '\tuse relative time'+ "\033[0m")
    else:
        print("\033[93m" + 'Use 2hop selections'+ "\033[0m")
    print("\033[93m" + 'num region '+ str(num_region) +  "\033[0m")
    print("\033[93m" + 'num msg '+ str(num_msg) +  "\033[0m")
    print("\033[93m" + 'window '+ str(window) +  "\033[0m")

    start = time.time()
    adv_nodes = [i for i in range(config.num_adv)]

    perigee = Experiment(
        node_hash,
        link_delay,
        node_delay,
        config.num_node,
        config.in_lim,
        out_lim, 
        num_region,
        data_path,
        adv_nodes,
        window
        )
    perigee.init_graph(outs_neighbors)

    perigee.start(max_epoch, record_epochs, num_msg)

def test_mf():
    if len(sys.argv) < 4:
        print('test-mf N L std num_exp')
        sys.exit(1)
    N = int(sys.argv[2])
    L = int(sys.argv[3])
    std = float(sys.argv[4])
    num = int(sys.argv[5])
    T = int(math.ceil(L * math.log(N)))
    mf_tester = tester.MF_tester(T, N, L, num)
    mf_tester.test_mf()

def test_mf_online():
    if len(sys.argv) < 9:
        print('test-mf seed[int] out[string] N[int] L[int] max_iter[int] new_msgs[int] std[float]')
        print('example: mf-online 1 N20_L5_iter20_msg5_std10_seed1 20 5 20 5 10')
        sys.exit(1)
    name = sys.argv[3]
    N = int(sys.argv[4])
    L = int(sys.argv[5])
    max_iter = int(sys.argv[6])
    num_msg = int(sys.argv[7])
    std = float(sys.argv[8])

    T = int(2*math.ceil(L * math.log(N)))
    online_mf_test = tester.MF_tester(T, N, L, max_iter, name)
    online_mf_test.online_test_mf(max_iter, num_msg, std)


def complete_graph():
    print('./testbed complete_graph 1 n 0 1 2 3 4 5')
    subcommand = sys.argv[1]
    data_path = sys.argv[3]
    out_lim = int(sys.argv[4])
    use_node_hash = sys.argv[5]=='y'

    record_epochs = [int(i) for i in sys.argv[6:]]
    max_epoch = max(record_epochs) +1
    num_region = out_lim

    [ node_delay, 
      node_hash, link_delay, 
      neighbor_set, IncomingLimit, 
      outs_neighbors, in_lims, 
      bandwidth] = initnetwork.GenerateInitialNetwork(
            config.network_type,
            config.num_node, 
            subcommand,
            out_lim)


    if out_lim != config.num_node-1:
        print('Error. A complete graph need correct out lim')
        sys.exit(1)
   
    outs_neighbors = {}
    for i in range(config.num_node):
        connections = []
        for j in range(config.num_node):
            if i != j:
                connections.append(j)
        outs_neighbors[i] = connections

    print('setup experiment')
    perigee = Experiment(
            node_hash,
            link_delay,
            node_delay,
            config.num_node,
            config.in_lim,
            out_lim, 
            data_path,
            [],
            0
            )
    perigee.init_graph(outs_neighbors)
    perigee.start_complete_graph(max_epoch, record_epochs)


if __name__ == '__main__':
    subcommand = sys.argv[1]
    if subcommand == 'test-mf':
        test_mf()
    elif subcommand == 'mf-online':
        test_mf_online()
    elif subcommand == 'complete_graph':
        complete_graph() 
    else:
        run()

