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
        self.out_lim = out_lim
        self.in_lim = in_lim
        self.best_count = 1
        self.state = state # states is the out conns
        self.num_rand = num_rand
        self.temperature = 1.5
        self.logger = logger
        self.logger.write_str('Init State: '+str(state))

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

    # currently use highest min
    def get_best_state(self, H, nodes, num_select):
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
                print('\t\tAdapt:Miss\t\tselected '+str(selected)+' rand '+str(conns))
            else:
                pools = set([i for i in range(self.num_node)])
                pools = pools.difference(set(selected))
                pools.remove(self.id)
                conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
                print('\t\tAdapt:Miss+random\t\tselected '+str(selected)+' rand '+str(conns))
            return selected, conns, True
            
    def set_state(self, outs):
        self.state = outs.copy()

    # nodes is a list whose value is node id
    def run_selector(self, H, nodes):
        T = int(H.shape[0] / 2)
        N = H.shape[1]
        # only consider the new msgs
        # H = H[T:]
        selected, rand_nodes, use_rand = self.get_best_state(H, nodes, self.out_lim)
        self.state = selected + rand_nodes
        self.best_count = 1
        if not use_rand:
            selected, rand_nodes = self.swap_rand_nodes(H, nodes)
        print('keep', selected)
        print('rand', rand_nodes)
        self.state = selected + rand_nodes
        return self.state
        

