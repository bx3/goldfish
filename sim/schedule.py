import numpy as np
import sys
import random
import itertools
import networkx as nx
import copy
from collections import namedtuple
from collections import defaultdict
import math 
import time

from numpy import linalg as LA

import config 

from utils import visualizer
from utils import comb_subset

from mat_complete import mat_comp_solver
from mat_factor import solver

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
        print('network_state denia')
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

def run_rel_comp(nodes, ld, nd, nh, optimizers, sparse_tables, bandits, update_nodes, time_tables, abs_time_tables, in_lim, out_lim, num_region, network_state, num_msg, pools, loggers, epoch, curr_conns, broad_nodes, topo_type, fixed_points):
    matrix_comp(
        optimizers, 
        sparse_tables,  
        bandits, 
        update_nodes, 
        num_msg, 
        pools, 
        loggers, 
        epoch,
        broad_nodes)


def run_mf(nodes, ld, nd, nh, optimizers, sparse_tables, bandits, update_nodes, time_tables, abs_time_tables, in_lim, out_lim, num_region, network_state, num_msg, pools, loggers, epoch, curr_conns, broad_nodes, topo_type, fixed_points):
    outs_neighbors = defaultdict(list)
    num_node = len(nodes)
    start = time.time()
    num_node_per_region = int((len(nodes)-1) / num_region)

    # run matrix factorization
    multithread_matrix_factor(
        optimizers, 
        sparse_tables,  
        bandits, 
        update_nodes, 
        num_msg, 
        pools, 
        loggers, 
        epoch,
        broad_nodes)

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
        loggers, 
        epoch, 
        curr_conns,
        broad_nodes, 
        num_node_per_region, 
        topo_type, 
        fixed_points, 
        ld, 
        nd
        )
    # print('*******')
    # print(epoch, outs_neighbors)
    for i, conns in curr_conns.items():
        if i not in update_nodes:
            outs_neighbors[i] = conns
    # print(epoch, outs_neighbors)

    return outs_neighbors

def process_WH(update_nodes, optimizers, bandits, sparse_tables, network_state, outs_neighbors, out_lim, num_msg, loggers, epoch, curr_conns, broad_nodes, num_node_per_region, topo_type, fixed_points, ld, nd):
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
        W_keep = bandits[i].set_ucb_table(W_reorder, X_input, mask_input, np.max(X_input))
        # update H mean table
        H_mean, sample_mean, H_mean_mask = get_H_mean(W_reorder, X_input, mask_input)
        opt.H_mean = H_mean.copy()
        opt.H_mean_mask = H_mean_mask.copy()

        opt = optimizers[i]
        window = opt.window
        window_W = broad_nodes[-window:]
        W_truth = np.zeros((opt.window, opt.L))
        if topo_type == 'dc': 
            for k in range(opt.window):
                b = int((window_W[k]-1)/num_node_per_region)
                W_truth[k,b] = 1
        elif topo_type == 'rand':
            if fixed_points is not None: 
                for k in range(opt.window):
                    b = fixed_points.index(window_W[k])
                    W_truth[k,b] =  1
        

        conns_str = ' '.join([str(j) for j in curr_conns[i]])
        conn_data_str = ' '.join([str(j) for j in np.sum(get_argmax_W(W_reorder), axis=0)])
        broad_window = broad_nodes[-opt.window:]
        
        num_show_mask = 10 
        num_showed = 10
        
        W_truth_reorder = match_W(W_reorder, W_truth.copy())
        # result_list = np.argmax(W_reorder, axis=1)== np.argmax(W_truth_reorder, axis=1)
        offs, next_hop_conns = evaluate_conns_on_fixed_miners(
            i, logger, curr_conns, broad_window[-num_showed:], nd, ld)

        logger.write_str('>'+str(epoch) + ' curr conn ' + conns_str  + ' X mask'  + ' num_data ' +conn_data_str) 
        logger.log_mats([logger.format_masked_mat(X_input[-num_show_mask:], mask_input[-num_show_mask:], False), 
            # logger.format_array(W_keep[-num_showed:]),
            logger.format_mat(W_reorder[-num_showed:],True), 
            logger.format_mat(get_argmax_W(W_reorder)[-num_showed:], False),
            # logger.format_mat(get_argmax_W(W_truth_reorder)[-num_showed:], False),
            # logger.format_mat(get_argmax_W(W_truth)[-num_showed:], False),
            # logger.format_array(result_list[-num_showed:]),
            logger.format_array(offs, '_Off_'),
            logger.format_array(next_hop_conns, '_Nhop'),
            logger.format_array(broad_window[-num_showed:], '_Src_')])
        # percent = sum(result_list)/float(len(result_list))
        mean_off = sum(offs)/len(offs)
        logger.write_str('>'+str(epoch)+' W_reorder W_chosen  W_truth. Mean offs: '+str(mean_off))

        # logger.write_str('X diff')
        # logger.log_mats([logger.format_masked_mat(X_input-W_reorder.dot(H_reorder), mask_input, False)])
        # logger.write_str('>'+str(epoch)+' H_reorder')
        # logger.log_mats([logger.format_masked_mat(H_reorder, H_mean_mask, False)])
        logger.write_str('>'+str(epoch)+' H_mean')
        logger.log_mats([logger.format_masked_mat(H_mean, H_mean_mask, False)])

        # pull arms
        scores, num_samples, max_time, bandit_T, score_mask = bandits[i].get_scores()
        

        valid_arms = get_pullable_arms(bandits[i].id, network_state)
        selected_arms = []
        bandit_arms = bandits[i].pull_arms(valid_arms, out_lim, window)
        # for l in range(out_lim):
            # if l not in keep_regions:
                # selected_arms.append(bandit_arms[l])
        peers = [p for j, p in sorted(bandit_arms, key=lambda x: x[0])]

        conns_str = ' '.join([str(j) for j in peers])       
        logger.write_str('>'+str(epoch)+' scores. next conn ' + conns_str)
        logger.log_mats([logger.format_masked_mat(scores, score_mask, True)])

        # update connections
        for p in peers:
            if is_connectable(i, p, network_state, []):
                outs_neighbors[i].append(p)
                network_state.add_in_connection(i, p)
            else:
                print('epoch', epoch, peers)
                print('bandit_arms', bandit_arms)
                # print('keep arm', keep_arms)
                print(i, p, 'not connectable')
                sys.exit(1)

def evaluate_conns_on_fixed_miners(u, logger, outs_conns, miners, nd, ld):
    G = initnetwork.construct_graph_by_outs_conns(outs_conns, nd, ld)
    offs = []
    next_hop_conns = []
    for m in miners:
        distance, path = nx.single_source_dijkstra(G, u, target=m)
        off = distance - ld[u, m]
        offs.append(off)
        next_hop_conns.append(path[1])
    return offs, next_hop_conns
    
# nh is node hash
def select_nodes(nodes, ld, num_msg, selectors, oracle, update_nodes, time_tables, in_lim, out_lim, network_state, num_keep, num_2hop, num_random):
    outs_neighbors = {} # output container
    num_invalid_compose = 0
    # direct peers
    num_rand_1hop = 0
    for i in update_nodes:
        keep_candidates = list(nodes[i].outs | nodes[i].ins )

        composes = comb_subset.get_config(
                num_keep, 
                keep_candidates,
                len(keep_candidates), 
                network_state,
                i)
        
        num_invalid_compose += math.comb(len(keep_candidates), num_keep) - len(composes)
        if len(composes) == 0:
            peers = selectors[i].select_random_peers(nodes, num_keep, network_state)
            num_rand_1hop += 1
            # oracle needs to know the connection
            oracle.update_1_hop_peers(i, peers)
            outs_neighbors[i] = peers
        else:
            for compose in composes:
                if len(compose) != len(set(compose)):
                    print('repeat in compose')
                    print(i)
                    print('composes', compose)
                    print(keep_candidates)
                    print('in', list(nodes[i].outs))
                    print('out', list(nodes[i].ins))
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
    # two hop peers
    if num_2hop > 0:
        for u in update_nodes:
            peers_info = oracle.get_multi_hop_info(u)
            peers, num_not_seen = selectors[u].select_peers(
                    config.num_2_hop, nodes, peers_info.two_hops, network_state)
            oracle.update_2_hop_peers(u, peers)
            outs_neighbors[u] += peers
            num_added_2hop += len(peers)

            tot_not_seen += num_not_seen
            
            # add 3hops
            if out_lim - len(outs_neighbors[u]) > num_random:
                num_3_hop = out_lim - len(outs_neighbors[u]) - num_random
                peers_info = oracle.get_multi_hop_info(u)
                peers, num_not_seen = selectors[u].select_peers(num_3_hop, nodes, peers_info.three_hops, network_state)
                oracle.update_3_hop_peers(u, peers)
                outs_neighbors[u] += peers
                num_added_3hop += len(peers) 
                tot_not_seen += num_not_seen
    
    # add random
    for u in update_nodes:
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


def matrix_comp(optimizers, sparse_tables, bandits, update_nodes, num_msg, pools, loggers, epoch, broad_nodes):
    args = []
    init_new = None

    for i in update_nodes:
        opt = optimizers[i]
        st = sparse_tables[i]
        # used with dc
        nodes_per_region = int((opt.N-1)/opt.L)

        arg = (i, st.table[-opt.window:], opt.N, opt.L)
        args.append(arg)

    results = pools.starmap(solver.run_pgd_nmf, args)

    assert(len(results) == len(update_nodes))
    for i in range(len(update_nodes)):
        W, H, penalties, j = results[i]
        # loggers[j].write_mat(H, '>' + str(epoch)+' H ' + str(opt))
        optimizers[j].store_WH(W, H)



def multithread_matrix_factor(optimizers, sparse_tables, bandits, update_nodes, num_msg, pools, loggers, epoch, broad_nodes):
    args = []
    start = time.time()
    init_new = None

    for i in update_nodes:
        opt = optimizers[i]
        st = sparse_tables[i]
        # used with dc
        nodes_per_region = int((opt.N-1)/opt.L)

        # construct grounth truth W
        window_W = broad_nodes[-opt.window:]
        W_truth = np.zeros((opt.window, opt.L))
        for k in range(opt.window):
            b = int((window_W[k]-1)/nodes_per_region)
            W_truth[k,b] = 1

        if opt.H_input is None or opt.W_input is None:
            # init W_input
            opt.W_input = W = np.random.uniform(0, 1, (opt.window, opt.L)) # W_truth.copy()
            # init H_input
            time_sum = 0
            num_t = 0
            for slot in st.table[-opt.window:]:
                for _, t in slot:
                    time_sum += t
                    num_t += 1
            X_mean = time_sum / num_t
            opt.H_input = np.random.uniform(0, X_mean, (opt.L, opt.N))
            logger = loggers[i] 
            logger.write_str('>'+str(epoch)+'init W truth, H_truth')
            logger.log_mats([logger.format_mat(opt.W_input, True)])
            logger.log_mats([logger.format_mat(opt.H_input, False)])
        else:
            # num_msg = len(st.table) - opt.W_prev.shape[0]
            # print("num_new_msg", num_msg, len(st.table), opt.W_prev.shape[0])
            opt.update_WH(num_msg)

        arg = (i, st.table[-opt.window:], opt.N, opt.L, opt.W_input, opt.H_input)
        args.append(arg)

    results = pools.starmap(solver.run_pgd_nmf, args)

    assert(len(results) == len(update_nodes))
    for i in range(len(update_nodes)):
        W, H, opt, j = results[i]
        # loggers[j].write_mat(H, '>' + str(epoch)+' H ' + str(opt))
        optimizers[j].store_WH(W, H)

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

def match_W(W_est, W_truth):
    truth_to_est, _ = get_est_mapping_col(W_est, W_truth)
    row, col = W_est.shape
    W_truth_reorder = np.zeros((row, col))
    for i in range(col):
        re_idx = truth_to_est[i]
        # print(re_idx, i)
        W_truth_reorder[:, re_idx] = W_truth[:, i]
    return W_truth_reorder


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


def get_est_mapping_col(H, H_est):
    col = H.shape[1]
    table = np.zeros((col, col))
    est_to_ori = {}
    for i in range(col):
        u = H[:,i]
        for j in range(col):
            v = H_est[:,j]
            dis = LA.norm(u-v) 
            table[i,j] = dis
    # print(table)
    est_to_ori, H_score = best_perm(table)
    # print(est_to_ori)

    # for i in range(len(H)):
        # est = np.argmin(table[i,:])
        # est_to_ori[est] = i 
    # print(m)
    # print(est_to_ori)
    return est_to_ori, H_score


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

def run_mc(completers, selectors, sparse_tables, network_state, epoch, update_nodes, pools):
    MC_get_HC(completers, sparse_tables, update_nodes, pools)
    conns = MC_select(completers, selectors, network_state, update_nodes)
    return conns

def MC_get_HC(completers, sparse_tables, update_nodes, pools):
    args = []
    for i in update_nodes:
        cpl = completers[i]
        st = sparse_tables[i]
        arg = (i, st.table[-cpl.T:], cpl.directions)
        args.append(arg)
    results = pools.starmap(mat_comp_solver.run, args)
    assert(len(results) == len(update_nodes))

    for i in range(len(update_nodes)):
        H, C, penalties, j, ids = results[i]
        completers[j].store_HC(H, C, ids, penalties)

    # debug
    for i in update_nodes:
        cpl = completers[i]
        st = sparse_tables[i]
        X, M, none_M, _, _ = mat_comp_solver.construct_table(st.table[-cpl.T:], i, cpl.directions)
        X_abs, M_abs, none_M_abs, _, ids_abs = mat_comp_solver.construct_table(st.abs_table[-cpl.T:], i, ['outgoing', 'incoming', 'bidirect'])

        completers[i].store_raw_table(X, M, none_M, st.broads[-cpl.T:], X_abs, M_abs, none_M_abs, ids_abs) # for debug



def MC_select(completers, selectors, network_state, update_nodes):
    outs_conns = {}
    for i in update_nodes:
        selector = selectors[i]
        completer = completers[i]
        outs = selector.run_selector(completer.H, completer.ids)
        outs_conns[i] = outs

    return outs_conns






