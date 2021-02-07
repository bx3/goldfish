import numpy as np
import sys
import random
import config 
import visualizer
from oracle import PeersInfo
import copy
import comb_subset
from collections import namedtuple
from collections import defaultdict
import math 
import time
import solver

# networkstate and oracle may look redundant, but oracle is used to answering 2hop
# networkstate is essential for checking if connection is possible
class NetworkState:
    def __init__(self, num_node, in_lim):
        self.num_in_conn = {} 
        self.in_conn_lim = {}
        self.conn = defaultdict(list)
        self.num_node = num_node
        for i in range(num_node):
            self.num_in_conn[i] = 0
            self.in_conn_lim[i] = in_lim

    def reset(self, num_node, in_lim):
        self.num_in_conn = {} 
        self.in_conn_lim = {}
        self.conn = defaultdict(list)
        for i in range(num_node):
            self.num_in_conn[i] = 0
            self.in_conn_lim[i] = in_lim

    def is_conn_addable(self, u, v):
        # if someone I want to connect, already connect me
        if u in self.conn[v]:
            #print(u, 'in', v)
            return False
        if v in self.conn[u]:
            #print(v, 'in', u)
            return False

        if self.num_in_conn[v] < self.in_conn_lim[v]:
            return True
        else:
            #print(v, 'in lim', u)
            return False

    def is_conn_keepable(self, u, v):
        # TODO
        if u in self.conn[v]:
            #print(u, 'in', v)
            return False
        if v in self.conn[u]:
            #print(v, 'in', u)
            return False

        if self.num_in_conn[v] < self.in_conn_lim[v]:
            return True
        else:
            #print(v, 'in lim', u)
            return False


    # v is the dst node 
    def add_in_connection(self, u, v):
        self.conn[u].append(v)
        self.num_in_conn[v] += 1

# selected by i
def is_connectable(i, p, network_state, selected):
    if p == i:
        return False 
    if p in selected:
        return False
    if network_state.is_conn_addable(i, p):
        return True
    else:
        return False

def get_pullable_arms(i, network_state):
    valid_arms = []
    for p in range(network_state.num_node):
        if p != i:
            if network_state.is_conn_addable(i, p):
                valid_arms.append(p)
    return valid_arms

def print_bandits(bandits):
    for i, bandit in bandits.items():
        arms = bandit.get_pulled_arms()
        print('\t\t*****', i)
        for a in arms:
            region, node = a
            print('node',i,'region',region,'peer',node, 'scores',bandit.ucb_table[a].score_list)

def get_random_regions(num, num_region):
    regions = [i for i in range(num_region)]
    random.shuffle(regions)
    return regions[:num]

def bandit_selection(bandit, W, X, network_state, outs_neighbors, out_lim, num_msg, max_time, logger, epoch, keeps):
    origins = bandit.update_ucb_table(W, X, num_msg, max_time)
    bandit.log_scores(logger, epoch, origins[-num_msg:])

    valid_arms = get_pullable_arms(bandit.id, network_state)
    # keep_arms = [a for l, a in keeps]
    # for a in keep_arms:
        # if a in valid_arms:
            # valid_arms.remove(a)

    arms = bandit.pull_arms(valid_arms, out_lim)


    # pulled_arms = bandit.get_pulled_arms()
    return arms

def select_nodes_by_matrix_completion(nodes, ld, nh, optimizers, sparse_tables, bandits, update_nodes, time_tables, abs_time_tables, in_lim, out_lim, num_region, network_state, num_msg, pools, loggers, epoch):
    outs_neighbors = defaultdict(list)
    num_node = len(nodes)

    start = time.time()
    # if config.num_thread == 1:
        # for i in update_nodes:
            # opt = optimizers[i]
            # st = sparse_tables[i]
            # W, H = solver.run_pgd_nmf(i, st.table[-st.window:], 
                    # st.N, st.L, opt.W, opt.H, st.window-num_msg)
            # X, max_time = solver.construct_table(st.N, st.table[-st.window:])
            # opt.store_WH(W, H)
            
            # peers = bandit_selection(
                    # bandits[i], W, X, network_state, 
                    # outs_neighbors, out_lim, num_msg, max_time, epoch)
            

            # # argmin_top_peers = choose_best_neighbor(H)
            # # argmin_peers = get_argmin_peers(i, H, network_state, outs_neighbors, out_lim)
            # # debug
            # # print(i, argmin_top_peers, argmin_peers)
            # # print(get_times(argmin_top_peers, X, out_lim))
            # # print(get_times(argmin_peers, X, out_lim))
            # # sys.exit(2)
            # for p in peers:
                # if is_connectable(i, p, network_state, outs_neighbors[i]):
                    # outs_neighbors[i].append(p)
                    # network_state.add_in_connection(i, p)

        # print('selection', round(time.time()-start, 2))
        # # print_bandits(bandits)
    # else:
    multithread_matrix_factor(nodes, optimizers, sparse_tables,  bandits, update_nodes, network_state, outs_neighbors, out_lim, num_region, num_msg, pools, loggers, epoch)

    # choose random peers
    # num_random = 0
    # start = time.time()
    # for i in update_nodes:
        # trial = 0
        # while len(outs_neighbors[i]) < out_lim:
            # num_random += 1
            # w = np.random.randint(num_node)
            # while not is_connectable(i, w, network_state, outs_neighbors[i]):
                # w = np.random.randint(num_node)
                # trial += 1
                # if trial == num_node-1:
                    # print(i, 'tried too many trial for random peer')
                    # break
            # outs_neighbors[i].append(w)
            # network_state.add_in_connection(i, w)
    return outs_neighbors
            

# nh is node hash
def select_nodes(nodes, ld, num_msg, nh, selectors, oracle, update_nodes, time_tables, in_lim, out_lim, network_state):
    outs_neighbors = {} # output container
    num_invalid_compose = 0
    # direct peers
    num_rand_1hop = 0
    for i in update_nodes:
        keep_candidates = list(nodes[i].outs)
        if config.both_in_and_out:
            keep_candidates += list(nodes[i].ins)       

        composes = comb_subset.get_config(config.num_keep, 
                keep_candidates,
                len(keep_candidates), 
                network_state,
                i)
        num_invalid_compose += math.comb(len(keep_candidates), config.num_keep) - len(composes)
        if len(composes) == 0:
            peers = selectors[i].select_random_peers(nodes, config.num_keep, network_state)
            num_rand_1hop += 1
            # oracle needs to know the connection
            oracle.update_1_hop_peers(i, peers)
            outs_neighbors[i] = peers
        else:
            for compose in composes:
                if len(compose) != len(set(compose)):
                    print(i)
                    print(compose)
                    print(list(nodes[i].outs))
                    print(list(nodes[i].ins))
                    sys.exit(1)

            peers = selectors[i].select_1hops(time_tables[i], composes, num_msg, network_state)
            # oracle needs to know the connection
            oracle.update_1_hop_peers(i, peers)
            outs_neighbors[i] = peers

    num_added_2hop = 0
    num_added_3hop = 0
    num_added_random = 0
    tot_not_seen = 0
    random.shuffle(update_nodes)
    print('shuffle to select 2hops')
    # two hop peers
    for u in update_nodes:
        peers_info = oracle.get_multi_hop_info(u)
        peers, num_not_seen = selectors[u].select_peers(
                config.num_2_hop, nodes, peers_info.two_hops, network_state)
        oracle.update_2_hop_peers(u, peers)
        outs_neighbors[u] += peers
        num_added_2hop += len(peers)

        tot_not_seen += num_not_seen
        
        # add 3hops
        if out_lim - len(outs_neighbors[u]) > config.num_random:
            num_3_hop = out_lim - len(outs_neighbors[u]) - config.num_random
            peers_info = oracle.get_multi_hop_info(u)
            peers, num_not_seen = selectors[u].select_peers(num_3_hop, nodes, peers_info.three_hops, network_state)
            oracle.update_3_hop_peers(u, peers)
            outs_neighbors[u] += peers
            num_added_3hop += len(peers) 
            tot_not_seen += num_not_seen
    
        # add random
        num_random = out_lim - len(outs_neighbors[u]) 
        num_added_random += num_random

        peers = selectors[u].select_random_peers(nodes, num_random, network_state)
        for p in peers:
            if p in outs_neighbors[u]:
                print(p, 'in neigbors', outs_neighbors[u])
                sys.exit(1)
        outs_neighbors[u] += peers

    # debug
    for u in update_nodes:
        if len(set(outs_neighbors[u])) != out_lim:
            print(u, "has less out neighbors")
            print(outs_neighbors[u])
            print(selectors[u].desc_conn)
            sys.exit(1)
    print('num_rand_1hop', num_rand_1hop,'num_invalid_compose', num_invalid_compose )
    # print('Finish. num2hop', num_added_2hop, 'num3hop', num_added_3hop, 'num rand', num_added_random, 'num no seen', tot_not_seen)
    return outs_neighbors

# simplest
def get_argmin_peers(i, H, network_state, outs_neighbors, out_lim):
    regions_order = [j for j in range(out_lim)]
    random.shuffle(regions_order)
    argsorted_peers = np.argsort(H, axis=1)
    arms = []
    for l in regions_order:
        peers = argsorted_peers[l, :]
        # choose one peers from that regions, peers are sorted in increasing time order
        for p in peers:
            if is_connectable(i, p, network_state, arms):
                arms.append(p)
                break
    return arms



def choose_best_neighbor(H):
    return np.argmin(H, axis=1)

def get_times(peers, X, out_lim):
    times = []
    for k in range(out_lim):
        p = peers[k]
        times.append(X[k, p])
    return times 

# debug
def print_mat(A):
    for i in range(A.shape[0]):
        text = ["{:4d}".format(int(a)) for a in A[i]]
        print(' '.join(text))

def multithread_matrix_factor(nodes, optimizers, sparse_tables, bandits, update_nodes, network_state, outs_neighbors, out_lim, num_region, num_msg, pools, loggers, epoch):
    args = []
    W_ = {}
    start = time.time()
    init_new = None

    for i in range(len(update_nodes)):
        opt = optimizers[i]
        st = sparse_tables[i]
        if opt.H is None:
            init_new = True
        else:
            init_new = False

        arg = (i, st.table[-st.window:], st.N, st.L, opt.W, opt.H, st.window-num_msg, init_new)
        args.append(arg)

    results = pools.starmap(solver.run_pgd_nmf, args)

    assert(len(results) == len(update_nodes))
    for i in range(len(update_nodes)):
        W, H, opt = results[i]

        loggers[i].write_mat(H, '>' + str(epoch)+' H ' + str(opt))

        W_[i] = W
        optimizers[i].store_WH(W, H)

    for i in update_nodes:
        opt = optimizers[i]
        st = sparse_tables[i]
        X, max_time = solver.construct_table(st.N, st.table[-opt.window:])

        selected_arms = []
        keep_arms = []
        keep_regions = [] # get_random_regions(config.num_untouch_arm, num_region) 
        # if config.num_untouch_arm > 0:
            # curr_peers = nodes[i].ordered_outs
            # if len(curr_peers) < out_lim:
                # print(i, 'curr_peers', curr_peers)
                # sys.exit(2)
            # keep_arms = select_keep_arms(i, curr_peers, keep_regions, network_state)
        # selected_arms += keep_arms

        # select arms
        bandit_arms = bandit_selection(
                bandits[i], W_[i], X, network_state, 
                outs_neighbors, out_lim, num_msg, max_time, loggers[i], epoch, keep_arms)
        
        for l in range(out_lim):
            if l not in keep_regions:
                selected_arms.append(bandit_arms[l])

        peers = [p for i, p in sorted(selected_arms, key=lambda x: x[0])]
            
        # update connections
        for p in peers:
            if is_connectable(i, p, network_state, outs_neighbors[i]):
                outs_neighbors[i].append(p)
                network_state.add_in_connection(i, p)
            else:
                print('epoch', epoch, peers)
                print('bandit_arms', bandit_arms)
                print('keep arm', keep_arms)
                print(i, p, 'not connectable')
                sys.exit(1)

def select_keep_arms(i, curr_peers, keep_regions, network_state):
    selected = []
    selected_nodes = []
    for l in keep_regions:
        k = curr_peers[l]
        while not is_connectable(i, k, network_state, selected_nodes):
            k = np.random.randint(network_state.num_node)
        selected.append((l, k))
        selected_nodes.append(k)
    # random.shuffle(keep_regions)
    # for l in keep_regions:
        # best_score, best_arm = None, []
        # for peer in curr_peers:
            # if peer not in selected_nodes:
                # score = bandit.get_ucb_score(l, peer)
                # if config.is_ucb:
                    # if best_score == None or score > best_score:
                        # best_score = score
                        # best_arm = [peer]
                    # elif score == best_score:
                        # best_arm.append(peer)
                # else:
                    # if best_score == None or score < best_score:
                        # best_score = score
                        # best_arm = [peer]
                    # elif score == best_score:
                        # best_arm.append(peer)
        # random.shuffle(best_arm)
        # arm = best_arm[0]
        # selected_nodes.add(arm)
        # selected.append((l, arm))
    return selected 
