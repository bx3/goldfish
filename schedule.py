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
from numpy import linalg as LA
import itertools


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

def run_mf(nodes, ld, nh, optimizers, sparse_tables, bandits, update_nodes, time_tables, abs_time_tables, in_lim, out_lim, num_region, network_state, num_msg, pools, loggers, epoch):

    outs_neighbors = defaultdict(list)
    num_node = len(nodes)
    start = time.time()
    # run matrix factorization
    multithread_matrix_factor(
        optimizers, 
        sparse_tables,  
        bandits, 
        update_nodes, 
        num_msg, 
        pools, 
        loggers, 
        epoch)

    # process arms
    process_WH(
        update_nodes, 
        optimizers, 
        bandits, 
        sparse_tables, 
        network_state, 
        outs_neighbors, 
        out_lim, 
        num_msg,
        loggers)

    return outs_neighbors

def process_WH(update_nodes, optimizers, bandits, sparse_tables, network_state, outs_neighbors, out_lim, num_msg, loggers):
    for i in update_nodes:
        opt = optimizers[i]
        st = sparse_tables[i]
        logger = loggers[i]
        X_input, mask_input, max_time = solver.construct_table(st.N, st.table[-opt.window:])
        # reorder H 
        _, _, W_reorder, H_reorder, _ = match_WH(opt.W_est, opt.H_est, opt.W_prev, opt.H_prev, False) 
        opt.H_prev = H_reorder.copy()
        opt.W_prev = W_reorder.copy()
        # update_ucb table
        bandits[i].set_ucb_table(W_reorder, X_input, mask_input, np.max(X_input))
        # update H mean table
        H_mean, sample_mean, H_mean_mask = get_H_mean(W_reorder, X_input, mask_input)
        opt.H_mean = H_mean.copy()
        opt.H_mean_mask = H_mean_mask.copy()

        opt = optimizers[i]
        logger.write_str('W_reorder W_chosen')
        logger.log_mats([
            logger.format_mat(W_reorder,True), 
            logger.format_mat(get_argmax_W(W_reorder), False)])
        logger.write_str('X mask')
        logger.log_mats([logger.format_masked_mat(X_input, mask_input, False)])
        logger.write_str('X diff')
        logger.log_mats([logger.format_masked_mat(X_input-W_reorder.dot(H_reorder), mask_input, False)])
        logger.write_str('H_reorder')
        logger.log_mats([logger.format_masked_mat(H_reorder, H_mean_mask, False)])
        logger.write_str('H_mean')
        logger.log_mats([logger.format_masked_mat(H_mean, H_mean_mask, False)])

        # pull arms
        # scores, num_samples, max_time, bandit_T, score_mask = bandits[i].get_scores()
        valid_arms = get_pullable_arms(bandits[i].id, network_state)
        selected_arms = []
        bandit_arms = bandits[i].pull_arms(valid_arms, out_lim)
        # for l in range(out_lim):
            # if l not in keep_regions:
                # selected_arms.append(bandit_arms[l])
        peers = [p for i, p in sorted(bandit_arms, key=lambda x: x[0])]


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

def multithread_matrix_factor(optimizers, sparse_tables, bandits, update_nodes, num_msg, pools, loggers, epoch):
    args = []
    start = time.time()
    init_new = None

    for i in range(len(update_nodes)):
        opt = optimizers[i]
        st = sparse_tables[i]
        
        if opt.H_input is None or opt.W_input is None:
            init_new = True
        else:
            init_new = False
            num_new_msg = len(st.table) - opt.W_prev.shape[0]
            opt.update_WH(num_new_msg)

        arg = (i, st.table[-opt.window:], opt.N, opt.L, opt.W_input, opt.H_input, init_new)
        args.append(arg)

    results = pools.starmap(solver.run_pgd_nmf, args)

    assert(len(results) == len(update_nodes))
    for i in range(len(update_nodes)):
        W, H, opt = results[i]
        loggers[i].write_mat(H, '>' + str(epoch)+' H ' + str(opt))
        optimizers[i].store_WH(W, H)

    # for i in update_nodes:
        # opt = optimizers[i]
        # st = sparse_tables[i]
        # X, max_time = solver.construct_table(st.N, st.table[-opt.window:])

        # selected_arms = []
        # keep_arms = []
        # keep_regions = [] # get_random_regions(config.num_untouch_arm, num_region) 
        # # if config.num_untouch_arm > 0:
            # # curr_peers = nodes[i].ordered_outs
            # # if len(curr_peers) < out_lim:
                # # print(i, 'curr_peers', curr_peers)
                # # sys.exit(2)
            # # keep_arms = select_keep_arms(i, curr_peers, keep_regions, network_state)
        # # selected_arms += keep_arms

        # # update_ucb table
        # bandits[i].set_ucb_table(W_[i], X, num_msg, np.max(X))
        # # update H mean table
        # H_mean, sample_mean, H_mean_mask = get_H_mean(W_reorder, X_input, mask_input)

        # # select arms
        # bandit_arms = bandit_selection(
                # bandits[i], W_[i], X, network_state, 
                # outs_neighbors, out_lim, num_msg, max_time, loggers[i], epoch, keep_arms)
        
        # for l in range(out_lim):
            # if l not in keep_regions:
                # selected_arms.append(bandit_arms[l])

        # peers = [p for i, p in sorted(selected_arms, key=lambda x: x[0])]
            
        # # update connections
        # for p in peers:
            # if is_connectable(i, p, network_state, outs_neighbors[i]):
                # outs_neighbors[i].append(p)
                # network_state.add_in_connection(i, p)
            # else:
                # print('epoch', epoch, peers)
                # print('bandit_arms', bandit_arms)
                # print('keep arm', keep_arms)
                # print(i, p, 'not connectable')
                # sys.exit(1)

def get_H_mean(W_est, X_input, mask):
    T = X_input.shape[0]
    N = X_input.shape[1]
    L = W_est.shape[1]

    H_mean_data = defaultdict(list)
    sum_sample= np.sum(X_input*mask) 
    num_sample = np.sum(mask) 
    for i in range(T):
        X_row = X_input[i]
        l = np.argmax(W_est[i])
        for j, t in enumerate(X_row):
            if mask[i,j] != 0:
                H_mean_data[(l,j)].append(t)
    sample_mean = sum_sample / num_sample
    H_mean = np.ones((L, N)) * sample_mean
    H_mean_mask = np.zeros((L, N))
    # for k, v in sorted(H_mean_data.items()):
        # print(k,np.round(v))
    for pair, samples in H_mean_data.items():
        l,j = pair
        H_mean[l,j] = sum(samples)/len(samples)   
        H_mean_mask[l,j] = 1
    return H_mean, sample_mean, H_mean_mask

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

def match_WH(W_est, H_est, W_ref, H_ref, is_compare):
    est_to_ori, H_score = get_est_mapping(H_ref, H_est)
    row, col = W_est.shape
    W_re = np.zeros((row, col))
    H_re = np.zeros(H_ref.shape)
    for i in range(col):
        re_idx = est_to_ori[i]
        W_re[:, re_idx] = W_est[:, i]
        H_re[re_idx] = H_est[i, :]
    W_score = -1
    if is_compare:
        W_score = compare_W(W_ref, W_re, {i: i for i in range(row)})
    return W_score, H_score, W_re, H_re, est_to_ori

def compare_W(W_ref, W_est, est_to_ori):
    W_re = np.zeros(W_est.shape)
    row, col = W_est.shape
    for i in range(col):
        re_idx = est_to_ori[i]
        W_re[:, re_idx] = W_est[:, i]

    a = np.argmax(W_ref, axis=1)
    b = np.argmax(W_re, axis=1)
    assert(len(a) == len(b))
    results = a == b
    W_score = sum(results)/len(results)
    return W_score


def best_perm(table):
    num_node = table.shape[0]
    nodes = [i for i in range(num_node)]
    best_comb = None
    best = 999999
    hist = {}
    for comb in itertools.permutations(nodes, num_node):
        score = 0
        # i is ori index, j in estimate index
        for i in range(num_node):
            j = comb[i]
            score += table[i,j]  
        hist[comb] = score
        if best_comb is None or score < best:
            best_comb = comb
            best = score
    est_to_ori = {}
    for i in range(num_node):
        est = best_comb[i]
        est_to_ori[est] = i

    return est_to_ori, best 


def get_est_mapping(H, H_est):
    table = np.zeros((len(H), len(H_est)))
    est_to_ori = {}
    for i in range(len(H)):
        u = H[i,:]
        for j in range(len(H_est)):
            v = H_est[j,:]
            dis = LA.norm(u-v) 
            table[i,j] = dis
    est_to_ori, H_score = best_perm(table)

    # for i in range(len(H)):
        # est = np.argmin(table[i,:])
        # est_to_ori[est] = i 
    # print(m)
    # print(est_to_ori)
    return est_to_ori, H_score

def get_argmax_W(W_est):
    chosen = np.argmax(W_est, axis=1)
    W_chosen = np.zeros(W_est.shape)
    for i, c in enumerate(chosen):
        W_chosen[i,c] = 1
    return W_chosen
