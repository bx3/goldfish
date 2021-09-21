#!/usr/bin/env python
import numpy as np
import sys
import random 
import torch
if len(sys.argv) >= 4:
    seed_num = int(sys.argv[2]) + 3
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    print("\033[93m" + 'random seed is set to '+ str(seed_num) + "\033[0m")

import time
import config
import math

from mat_factor import tester
from sec_hop import selector
from experiment import Experiment
import matplotlib.pyplot as plt
import net_init
from simple_model.simple_model import NetworkSim
from sec_hop.experiment import Experiment as SecHopExperiment

# assuming uniform, num_pub => num region, if this info is known
# otherwise use a large number
def calculate_T(miss_prob, num_node, num_pub, num_topo, top_n_peer):
    T = math.ceil(math.log(miss_prob) / math.log(1.0 - 1.0/num_pub)) / top_n_peer
    return int(T * num_topo*top_n_peer)

num_out = 4
num_in =  8
num_2hop = 0
num_rand = 1
num_msg = 40 # 20 *2   

def run_simple_model():
    subcommand = sys.argv[1]
    seed = int(sys.argv[2])
    topo = sys.argv[3]
    plt_name = sys.argv[4]
    num_star = int(sys.argv[5])
    num_epoch = int(sys.argv[6])
    churn_rate = float(sys.argv[7])
    print_log = sys.argv[8]=='y'


    num_topo = 3 
    update_interval = 2

    top_n_peer = 2
    # extra info, extra choose appropriate T as hyperparameter
    num_pub, num_node, pubs = net_init.get_num_pub_node(topo)
    # T = calculate_T(0.001, num_node, num_pub, num_topo, top_n_peer) 
    T = num_msg*num_topo
    print('pubs', pubs)

    # pools = [i for i in range(num_node) if i not in pubs] # node using adaptive algo 
    pools = [i for i in range(num_node)]
    stars = list(np.random.choice(pools, num_star, replace=False))
    print('stars', stars)

    mc_epochs = 2000
    mc_lr = 1e-2
    mc_exit_loss_diff = 1e-3

    
    m = NetworkSim(topo, num_out, num_in, num_epoch, T, num_topo, stars, mc_epochs, mc_lr, mc_exit_loss_diff, num_rand, top_n_peer, plt_name, print_log, churn_rate, update_interval)
    start_t = time.time()
    m.run()
    print('finish', num_epoch, 'epochs in', round(time.time()-start_t,2), 'sec')



def run_mc():
    subcommand = sys.argv[1]
    seed = int(sys.argv[2])
    data_path = sys.argv[3]
    out_lim = int(sys.argv[4])
    num_new = int(sys.argv[5])
    input_json = sys.argv[6]
    use_logger = sys.argv[7] == 'y'
    loc, link_delay, role, proc_delay = net_init.load_network(input_json)
    record_epochs = [int(i) for i in sys.argv[8:]]
    max_epoch = max(record_epochs) +1
    num_node = len(loc)
    num_msg =  out_lim*3 *2 # assume numIn = numOut, num include both
    window = 2*num_msg
    in_lim = num_node

    print("\033[93m" + 'num msg per topo '+ str(num_msg) +  "\033[0m")
    print("\033[93m" + 'mat complete window size'+ str(window) +  "\033[0m")

    start = time.time()
    adv_nodes = [i for i in range(config.num_adv)]
    perigee = Experiment(
        link_delay,
        role,
        proc_delay,
        num_node,
        in_lim,
        out_lim, 
        data_path,
        adv_nodes,
        window,
        'mc',
        loc,
        num_new,
        use_logger,
        seed
        )
    perigee.init_graph_mc()
    perigee.start(max_epoch, record_epochs, num_msg)


def run_mf():
    subcommand = sys.argv[1]
    data_path = sys.argv[3]
    out_lim = int(sys.argv[4])
    num_region = int(sys.argv[5])
    use_node_hash = sys.argv[6]=='y'
    input_json = sys.argv[7]

    assert(config.use_abs_time)
    loc, link_delay = net_init.load_network(input_json)

    record_epochs = [int(i) for i in sys.argv[8:]]
    max_epoch = max(record_epochs) +1
    window = int(config.num_msg*config.window_constant * math.ceil(num_region * math.log(config.num_node))) # T > L log N
    
    print(window, num_region, config.num_node)
    node_delay = net_init.GenerateInitialDelay(config.num_node)
    node_hash = None
    # [LinkDelay,NodeHash,NodeDelay] = readfiles.Read(NodeDelay, NetworkType, num_node)

    # [ node_delay, 
      # node_hash, _, 
      # _, _, 
      # _, _, 
      # _] = initnetwork.GenerateInitialNetwork(
            # config.network_type,
            # config.num_node, 
            # subcommand,
            # out_lim)
    # tester.print_mat(link_delay, False)

    if config.use_reduce_link:
        print("\033[91m" + 'Use reduced link latency' + "\033[0m")
        net_ini.reduce_link_latency(config.num_node, int(0.2*config.num_node), link_delay)
    else:
        print("\033[93m" + 'Not use reduced link latency'+ "\033[0m")

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
    print("\033[93m" + 'num msg '+ str(config.num_msg) +  "\033[0m")
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
        window,
        'mf-bandit' ,
        loc
        )
    perigee.init_graph()
    perigee.start(max_epoch, record_epochs, config.num_msg)

def run_rel_comp():
    subcommand = sys.argv[1]
    data_path = sys.argv[3]
    out_lim = int(sys.argv[4])
    num_region = int(sys.argv[5])
    use_node_hash = sys.argv[6]=='y'
    input_json = sys.argv[7]

    loc, link_delay = net_init.load_network(input_json)

    # record_epochs = [int(i) for i in sys.argv[8:]]
    # max_epoch = max(record_epochs) +1
    # window = int(config.num_msg*config.window_constant * math.ceil(num_region * math.log(config.num_node))) # T > L log N
    
    # print(window, num_region, config.num_node)
    # node_delay = initnetwork.GenerateInitialDelay(config.num_node)
    # node_hash = readfiles.ReadHashFile(config.num_node)
    # adv_nodes = []
    # assert(config.use_abs_time == False)
    # perigee = Experiment(
        # node_hash,
        # link_delay,
        # node_delay,
        # config.num_node,
        # config.in_lim,
        # out_lim, 
        # num_region,
        # data_path,
        # adv_nodes,
        # window,
        # 'rel_comp' ,
        # loc
        # )
    # perigee.init_graph()
    # perigee.start_rel_comp(max_epoch, record_epochs, config.num_msg)


def run_2hop():
    seed = int(sys.argv[2])

    topo =  sys.argv[3]
    num_epoch = int(sys.argv[4])
    num_adapt = int(sys.argv[5])
    churn_rate = float(sys.argv[6])
    data_path = sys.argv[7]


    num_pub, num_node, pubs = net_init.get_num_pub_node(topo)



    num_keep = num_out - num_rand - num_2hop

    print('pubs', pubs)
    # pools = [i for i in range(num_node) if i not in pubs] # node using adaptive algo 
    pools = [i for i in range(num_node)] # node using adaptive algo 

    stars = list(np.random.choice(pools, num_adapt, replace=False))
    print('stars', stars)

    sec_hop_model = SecHopExperiment(
        topo,
        num_in,
        num_out, 
        data_path,
        num_keep, 
        num_2hop, 
        num_rand,
        num_epoch,
        stars,
        num_msg,
        churn_rate,
        )
    sec_hop_model.run()

def complete_graph():
    print('./testbed complete_graph 1 n 0 1 2 3 4 5')
    subcommand = sys.argv[1]
    data_path = sys.argv[3]
    out_lim = int(sys.argv[4])
    use_node_hash = sys.argv[5]=='y'

    record_epochs = [int(i) for i in sys.argv[6:]]
    max_epoch = max(record_epochs) +1
    num_region = out_lim

    # [ node_delay, 
      # node_hash, link_delay, 
      # neighbor_set, IncomingLimit, 
      # outs_neighbors, in_lims, 
      # bandwidth] = initnetwork.GenerateInitialNetwork(
            # config.network_type,
            # config.num_node, 
            # subcommand,
            # out_lim)

    # if out_lim != config.num_node-1:
        # print('Error. A complete graph need correct out lim')
        # sys.exit(1)
   
    # outs_neighbors = {}
    # for i in range(config.num_node):
        # connections = []
        # for j in range(config.num_node):
            # if i != j:
                # connections.append(j)
        # outs_neighbors[i] = connections

    # print('setup experiment')
    # perigee = Experiment(
            # node_hash,
            # link_delay,
            # node_delay,
            # config.num_node,
            # config.in_lim,
            # out_lim, 
            # data_path,
            # [],
            # 0,
            # subcommand
            # )
    # perigee.init_graph(outs_neighbors)
    # perigee.start_complete_graph(max_epoch, record_epochs)

def test_mf():
    if len(sys.argv) < 4:
        print('test-mf N L std num_exp')
        sys.exit(1)
    # N = int(sys.argv[2])
    # L = int(sys.argv[3])
    # std = float(sys.argv[4])
    # num = int(sys.argv[5])
    # T = int(math.ceil(L * math.log(N)))
    # mf_tester = tester.MF_tester(T, N, L, num)
    # mf_tester.test_mf()

def parse_static_input():
    if len(sys.argv) < 13:
        print('Error. Invalud arguments')
        print('test-mf seed[int] out[string] N[int] L[int] max_iter[int] new_msgs[int] noise[float] H_method[unif,log-unif,1D-linear] add_method[rolling,append] num_mask[int] init_method[algo,ref] mask_method[random,bandit] windowM[float]')
        print('example: ./testbed.py mf-static 1 node=20 region=5 new_msgs=1 H_method=1D-linear add_method=append mask_method=bandit init_method=ref name=rand-init num_mask=0 noise=0 max_iter=100 windowM=4 ')
        sys.exit(1)

    for arg in sys.argv[3:]:
        if 'noise=' in arg:
            std = float(arg[6:])
        elif 'node=' in arg:
            N = int(arg[5:])
        elif 'region=' in arg:
            L = int(arg[7:])
        elif 'new_msgs=' in arg:
            num_msg = int(arg[9:])
        elif 'H_method=' in arg:
            H_method = arg[9:]
        elif 'add_method=' in arg:
            add_method = arg[11:]
        elif 'max_iter=' in arg:
            max_iter = int(arg[9:])
        elif 'num_mask=' in arg:
            num_mask_per_row = int(arg[9:])
        elif 'mask_method=' in arg:
            mask_method = arg[12:]
        elif 'name=' in arg:
            name = arg[5:]
        elif 'init_method=' in arg:
            init_method = arg[12:]
        elif 'windowM=' in arg:
            windowM = float(arg[8:])
        else:
            print('Error. Unknown arg', arg)
            sys.exit(1)

    return (std, N, L, num_msg, H_method, add_method, max_iter, num_mask_per_row, mask_method, name, init_method, windowM)

def run_1hop_static():
    (std, N, L, num_msg, H_method, 
     add_method, max_iter, num_mask_per_row, 
     mask_method, name, init_method, windowM) =  parse_static_input()

    exp_name = ('static1hop_node'+str(N)+'-'+'region'+str(L)+"-"+'noise'+str(int(std))+'-'+
            H_method+'-'+add_method+str(num_msg)+'msg'+'-'+mask_method+'-'+str(num_mask_per_row)+
            'mask'+'-'+init_method+'-'+'windowM'+ str(int(windowM))+'-'+name)

    T = int(windowM*math.ceil(L * math.log(N))) # 3
    static1hop_exp = tester.MF_tester(T, N, L, max_iter, exp_name, add_method, num_mask_per_row, 
            H_method, init_method, mask_method)
    static1hop_exp.run_1hop(max_iter, num_msg, std)

def run_mf_static():
    (std, N, L, num_msg, H_method, 
     add_method, max_iter, num_mask_per_row, 
     mask_method, name, init_method, windowM) =  parse_static_input()

    exp_name = ('node'+str(N)+'-'+'region'+str(L)+"-"+'noise'+str(int(std))+'-'+
            H_method+'-'+add_method+str(num_msg)+'msg'+'-'+mask_method+'-'+str(num_mask_per_row)+
            'mask'+'-'+init_method+'-'+'windowM'+ str(int(windowM))+'-'+name)
    print('exp_name', exp_name)

    T = int(windowM*math.ceil(L * math.log(N))) # 3
    mf_online_exp = tester.MF_tester(T, N, L, max_iter, exp_name, add_method, num_mask_per_row, 
            H_method, init_method, mask_method)
    W_scores, H_scores = mf_online_exp.start_mf_online(max_iter, num_msg, std)
    with open(mf_online_exp.result_filepath, 'w') as w:
        # w.write(str(int(std)) + '\t' + str(num_mask_per_row) + '\n')
        w.write('\t'.join([str(a) for a in W_scores]) + '\n')
        w.write('\t'.join([str(a) for a in H_scores]) + '\n')

    iters = [i for i in range(len(W_scores))]
    fig, axs = plt.subplots(2)
    axs[0].scatter(iters, W_scores, c='r', s=10)
    axs[0].set_title(exp_name + '\n'+'W best perm score')
    axs[1].scatter(iters, H_scores, s=10)
    axs[1].set_title('H best perm score')
    plt.tight_layout()
    plt.show()
    fig.savefig(mf_online_exp.fig_filepath)

def get_window_size(num_mask_per_row, N, L):
    pass

def print_help():
    print('subcommand')
    print('\trun-2hop       seed[int] output_dir[str] num_out[int] num_msg[int] use_node_hash[y/n] rounds[intList]')
    print('\trun-mf         seed[int] output_dir[str] num_out[int] num_region[int] use_node_hash[y/n] rounds[intList]')
    print('\trun-mc         seed[int] output_dir[str] num_out[int] num_new[int] input_json[str] rounds[intList]')

    print('\trel-comp         seed[int] output_dir[str] num_out[int] num_region[int] use_node_hash[y/n] rounds[intList]')

    print('\tcomplete-graph seed[int] output_dir[str] num_out[int] num_region[int] use_node_hash[y/n] 1')
    print('\tmf-static      seed[int] help')
    print('\t1hop-static    seed[int] help')
    print('\ttest_mf        N[int] L[int] std[float] num_exp[int]')
    print('Note. Output is stored at analysis/output_dir')


if __name__ == '__main__':
    subcommand = sys.argv[1]
    if subcommand == 'help':
        print_help()
    elif subcommand == 'test-mf':
        test_mf()
    elif subcommand == 'mf-static':
        run_mf_static()
    elif subcommand == 'complete_graph':
        complete_graph() 
    elif subcommand == 'run-2hop':
        run_2hop()
    elif subcommand == 'run-mf':
        run_mf()
    elif subcommand == 'rel-comp':
        run_rel_comp()
    elif subcommand == '1hop-static':
        run_1hop_static()
    elif subcommand == 'run-mc':
        run_mc()
    elif subcommand == 'run-simple-model':
        run_simple_model()
    else:
        print('Error. Unknown subcommand', subcommand)
        print_help()
        sys.exit(1)
