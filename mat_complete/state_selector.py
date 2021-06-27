import numpy as np
from collections import defaultdict
from mat_complete.mat_comp_solver import print_mats
from mat_complete.mat_comp_solver import format_mat
from mat_complete.mat_comp_solver import format_array
import random


class StateSelector:
    def __init__(self, i, num_node, out_lim, in_lim, state, num_rand, logger):
        self.id = i
        self.num_node = num_node
        self.out_lim = out_lim
        self.in_lim = in_lim
        self.best_count = 1
        self.state = state # states is the out conns
        self.num_rand = num_rand
        self.temperature = 1
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
        for i in range(self.num_node):
            if i != self.id and i not in self.state and i not in nodes:
                pools.append(i)
        rand_nodes = list(np.random.choice(pools, self.num_rand, replace=False))
        # get good nodes to retain
        selected = [] 
        ranks = self.get_ranks(H) 
        sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        for i in range(self.out_lim-self.num_rand):
            ind = sorted_ranks[i][0]
            node_id = nodes[ind]
            selected.append(node_id)

        # new_state = selected + rand_nodes

        named_ranks = [(nodes[n[0]], n[1]) for n in sorted_ranks]
        return selected, rand_nodes, named_ranks

    def get_ranks(self, H):
        # TODO it is possibel two column for a row has the same lowest entry, ignore now
        row_mins = np.argmin(H, axis=1) 
        ranks = defaultdict(int)
        for r in row_mins:
            ranks[r] += 1
        return ranks

    # currently use highest min
    def get_best_state(self, H, nodes, num_select, broads):
        T = H.shape[0]
        N = H.shape[1]
        ranks = self.get_ranks(H) 
        sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        selected = []
        if len(sorted_ranks) >= num_select:
            for i in range(num_select):
                ind = sorted_ranks[i][0]
                node_id = nodes[ind]
                selected.append(node_id)
            # print('>', self.id, selected) 
            # print('node', self.id)
            # print_mats([format_mat(H, nodes, False), format_array(broads, 'src')])
            # print(ranks)
            # print(selected)
            # sys.exit(1)
            return selected
        else:
            for i in range(len(sorted_ranks)):
                ind = sorted_ranks[i][0]
                node_id = nodes[ind]
                selected.append(node_id)
            # in the case, when number less than num_select() nodes are best
            # i.e. the other nodes are strictly worse in all rows
            # choose a random
            pools = set([i for i in range(self.num_node)])
            pools = pools.difference(set(nodes))
            pools.remove(self.id)
            conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
            self.logger.write_str('\t\tAdapt:Miss\t\tselected '+str(selected)+' rand '+str(conns))
            return selected + conns
            


    # nodes is a list whose value is node id
    def run_selector(self, H, nodes, broads):
        curr_state = self.get_best_state(H, nodes, self.out_lim, broads)
        # self.logger.write_str('\t\tAdapt: get new state '+str(sorted(curr_state)))
        if sorted(curr_state) != sorted(self.state):
            self.logger.write_str('\t\tAdapt:Diff\t\tstate_new '+str(sorted(curr_state))+ ' != state_old '+str(sorted(self.state)))
            self.state = curr_state
            self.best_count = 1
        else:
            self.best_count += 1
            if self.is_random():
                # TODO get rand peers
                self.best_count = 1
                selected, rand_nodes, ranks = self.swap_rand_nodes(H, nodes)
                self.logger.write_str('\t\tAdapt:Same\t\tSwap: '+str(self.state)+'->'+str(selected+ rand_nodes)+'. Keep'+str(selected)+' For '+str(ranks))
                self.state = selected+ rand_nodes
            else:
                self.logger.write_str('\t\tAdapt:Same\t\tKeep: '+str(self.state))
        return self.state
        

