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

# networkstate and oracle may look redundant, but oracle is used to answering 2hop
# networkstate is essential for checking if connection is possible
class NetworkState:
    def __init__(self, num_node, in_lim):
        self.num_in_conn = {} 
        self.in_conn_lim = {}
        self.conn = defaultdict(list)
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

def bandit_selection(i, argsorted_peers, network_state, outs_neighbors, out_lim):
    # select for each outo
    assert(out_lim == argsorted_peers.shape[0])
    regions_order = [i for i in range(out_lim)]
    random.shuffle(regions_order)
    for l in regions_order:
        peers = argsorted_peers[l, :]
        # choose one peers from that regions, peers are sorted in increasing time order
        for p in peers:
            if is_connectable(i, p, network_state, outs_neighbors[i]):
                outs_neighbors[i].append(p)
                network_state.add_in_connection(i, p)
                break

def select_nodes_by_matrix_completion(nodes, ld, nh, optimizers, bandits, update_nodes, time_tables, abs_time_tables, in_lim, out_lim, network_state, pools):
    outs_neighbors = defaultdict(list)
    num_node = len(nodes)

    if config.num_thread == 1:
        start = time.time()
        for i in update_nodes:
            argsorted_peers = optimizers[i].matrix_factor()
            peers = bandit_selection(i, argsorted_peers, network_state, outs_neighbors, out_lim)
        print('selection', round(time.time()-start, 2))
    else:
        futures = []
        for i in update_nodes:
            future = pools.submit(optimizers[i].matrix_factor(), None)
            futures.append(future)

        for future in futures:
            peers = future.result()
            for p in peers:
                if is_connectable(i, p, network_state, outs_neighbors[i]):
                    outs_neighbors[i].append(p)
                    network_state.add_in_connection(i, p)

    # choose random peers
    num_random = 0
    start = time.time()
    for i in update_nodes:
        trial = 0
        while len(outs_neighbors[i]) < out_lim:
            num_random += 1
            w = np.random.randint(num_node)
            while not is_connectable(i, w, network_state, outs_neighbors[i]):
                w = np.random.randint(num_node)
                trial += 1
                if trial == num_node-1:
                    print(i, 'tried too many trial for random peer')
                    break
            outs_neighbors[i].append(w)
            network_state.add_in_connection(i, w)
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
