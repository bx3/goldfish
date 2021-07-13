import numpy as np
from collections import defaultdict
from mat_complete.mat_comp_solver import print_mats
from mat_complete.mat_comp_solver import format_mat
from mat_complete.mat_comp_solver import format_array
import random
import sys


class SimpleSelector:
    def __init__(self, i, num_node, out_lim, in_lim, state, num_rand, logger):
        self.id = i
        self.num_node = num_node
        self.known_peers = [j for j in range(self.num_node) if i != j]
        self.out_lim = out_lim
        self.in_lim = in_lim
        self.best_count = 1
        self.state = state # states is the out conns
        self.num_rand = num_rand
        self.temperature = 1.5
        self.logger = logger
        self.logger.write_str('Init State: '+str(state))

        self.hist_explored_peers = {} # key is peer, value is the last time in mc table
        self.counter = 0

    def is_random(self):
        prob = 1 - (1.0/float(self.best_count))**self.temperature
        if random.random() < prob:
            return False
        else:
            return True

    def swap_rand_nodes(self, H, nodes):
        pools = []

        # get good nodes to retain
        selected = [] 
        ranks = self.get_ranks(H, nodes) 
        sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        for i in range(self.out_lim-self.num_rand):
            node_id = sorted_ranks[i][0]
            if node_id == self.id:
                print(self.id, 'selected itself in swap')
                sys.exit(1)
            selected.append(node_id)

        for i in range(self.num_node):
            if i != self.id and i not in self.state and i not in nodes:
                pools.append(i)

        if len(pools) < self.num_rand:
            for i in range(self.num_node):
                if i != self.id and i not in self.state:
                    pools.append(i)

            if i != self.id and i not in self.state and i not in nodes:
                pools.append(i)
            pools = set([i for i in range(self.num_node)])
            pools = pools.difference(set(selected))
            pools.remove(self.id)
            rand_nodes = list(np.random.choice(list(pools), self.out_lim-len(selected), replace=False))
            print('pool size not enough', pools, self.state, nodes)
        else:
            rand_nodes = list(np.random.choice(pools, self.num_rand, replace=False))
        
        # named_ranks = [(nodes[n[0]], n[1]) for n in sorted_ranks]
        return selected, rand_nodes

    def get_ranks(self, H, node_ids):
        # TODO it is possibel two column for a row has the same lowest entry, ignore now
        row_mins = np.argmin(H, axis=1) 
        ranks = defaultdict(int)
        for r in row_mins:
            index = node_ids[r]
            ranks[index] += 1
        return ranks

    def get_rand_peers(self, curr_peers, keep_peers, num_explore):
        for p in curr_peers + keep_peers:
            self.hist_explored_peers[p] = self.counter
        pools = set(self.known_peers)
        cands = pools.difference(self.hist_explored_peers.keys())
        print(self.counter, 'seen', len(self.hist_explored_peers), ' peers', sorted(list(self.hist_explored_peers.keys())))
        if len(cands) >= num_explore:
            explores = list(np.random.choice(list(cands), num_explore, replace=False))
            print('\t\tAdapt:\t\texplore  '+str(sorted(explores)))
            return explores
        else:
            explores = list(cands)
            num_explore -= len(cands)
            self.hist_explored_peers.clear()
            new_pool_explore = self.get_rand_peers(curr_peers, keep_peers, num_explore)
            print('\t\tAdapt:\t\texplore  '+str(sorted(explores+new_pool_explore)))
            return explores + new_pool_explore

    # currently use highest min
    def get_best_ranks(self, H, nodes, num_select):
        T = H.shape[0]
        N = H.shape[1]       
        ranks = self.get_ranks(H, nodes) 
        sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        print('sorted_ranks', num_select, sorted_ranks)
        selected = []
        if len(sorted_ranks) >= num_select:
            for i in range(num_select):
                node_id = sorted_ranks[i][0]
                selected.append(node_id)
            print('\t\tAdapt:\t\tselected '+str(selected))
            return selected, [], False
        else:
            for i in range(len(sorted_ranks)):
                node_id = sorted_ranks[i][0]
                selected.append(node_id)

            pools = set([i for i in range(self.num_node)])
            pools = pools.difference(set(nodes))
            pools.remove(self.id)
            if len(pools) >= num_select-len(selected):
                conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
                print('\t\tAdapt:Miss\t\tselected '+str(sorted(selected))+' rand '+str(conns))
            else:
                pools = set([i for i in range(self.num_node)])
                pools = pools.difference(set(selected))
                pools.remove(self.id)
                conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
                print('\t\tAdapt:Miss+random\t\tselected '+str(sorted(selected))+' rand '+str(sorted(conns)))
            return selected, conns, True
            
    def set_state(self, outs):
        self.state = outs.copy()

    # nodes is a list whose value is node id
    def run_selector(self, H_in, nodes, non_value_mask):
        # select node to keep
        H = H_in.copy()*(1-non_value_mask) + 1e10*non_value_mask
        selected, rand_nodes, use_rand = self.get_best_ranks(H, nodes, self.out_lim - self.num_rand)
        # select node to explore
        self.state = selected + rand_nodes

        explore_nodes = self.get_rand_peers(nodes, selected + rand_nodes, self.num_rand)

        self.state = selected + rand_nodes + explore_nodes
        self.counter += 1
        return selected + rand_nodes, explore_nodes
        

