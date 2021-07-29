import numpy as np
from collections import defaultdict
from simple_model import explorer
from simple_model import formatter
from simple_model import employer
import random
import sys
# import itertools

class SimpleSelector:
    def __init__(self, i, num_node, out_lim, in_lim, state, num_rand, logfile):
        self.id = i
        self.num_node = num_node
        self.known_peers = [j for j in range(self.num_node) if i != j]
        self.out_lim = out_lim
        self.in_lim = in_lim
        self.state = state # states is the out conns
        self.num_rand = num_rand
        self.depleting_pool = explorer.DepletingPool(self.id, self.known_peers, logfile)
        self.greedy_explorer = explorer.GreedyExplorer(i)

        self.subset_exploiter = employer.SubsetExploiter(logfile, self.id)
        self.count_exploiter = employer.CountExploiter(logfile, self.id)

        self.log = logfile

    def find_x_rand_to_conn(self, pools, num, oracle):
        conns = []
        for i in pools:
            if len(oracle.can_i_connect(self.id, [i])) == 0:
                conns.append(i)
            if num == len(conns):
                break
        if num != len(conns):
            print('cannot find ', num, 'peers satisfying oracle')
            sys.exit(1)
        return conns

    def draw_random_peers(self, excludes, num, oracle):
        pools = set([i for i in range(self.num_node)])
        pools = pools.difference(set(excludes))
        pools.remove(self.id)

        conns = []
        if len(pools) >= num:
            # conns = list(np.random.choice(list(pools), num, replace=False))
            pools = list(pools)
            np.random.shuffle(pools)
            conns = self.find_x_rand_to_conn(pools, num, oracle)
            formatter.printt('\t\tExploit(random):\t\t{}\n'.format(conns), self.log)
        else:
            pools = set([i for i in range(self.num_node)])
            pools = pools.difference(set(selected))
            pools.remove(self.id)
            np.random.shuffle(pools)
            conns = self.find_x_rand_to_conn(pools, num, oracle)

            formatter.printt('\t\tExploit(random+reuse):\t\t{}\n'.format(conns), self.log)
        return conns

    def set_state(self, outs):
        self.state = outs.copy()

    def remove_H_non_value_rows(self, H_in, unkn_plus_mask, plus_mask, unkn_unab_mask):
        T, N = H_in.shape
        non_value_mask = 1*((plus_mask+unkn_unab_mask+unkn_plus_mask)>0)

        non_value_rows = []
        for i in range(T):
            if np.sum(non_value_mask[i]) == N:
                non_value_rows.append(i)

        H = np.delete(H_in, tuple(non_value_rows), axis=0)
        plus_mask = np.delete(plus_mask, tuple(non_value_rows), axis=0)
        unkn_unab_mask = np.delete(unkn_unab_mask, tuple(non_value_rows), axis=0)
        unkn_plus_mask = np.delete(unkn_plus_mask, tuple(non_value_rows), axis=0)

        return H, unkn_plus_mask, plus_mask, unkn_unab_mask

    def run_selector(self, H_in, nodes, unkn_plus_mask, plus_mask, unkn_unab_mask, oracle, curr_out):
        non_value_mask = 1*((plus_mask+unkn_unab_mask+unkn_plus_mask)>0)
        H = H_in.copy()*(1-non_value_mask) + 1e10*non_value_mask
        H, unkn_plus_mask, plus_mask, unkn_unab_mask = self.remove_H_non_value_rows(H, unkn_plus_mask, plus_mask, unkn_unab_mask)

        # formatter.print_unformated_mat(H, unkn_plus_mask, plus_mask, unkn_unab_mask)
        formatter.printt('\tExploit vs. Explore\n', self.log)
        num_select = self.out_lim-self.num_rand

        rand_nodes = []
        selected = self.subset_exploiter.select_subset_peer(H, nodes,num_select,plus_mask,oracle,curr_out)
        if len(selected) != num_select:
             # selected = self.count_exploiter.select_best_peer(H, nodes, num_select, plus_mask, oracle)
             # if len(selected) != num_select:
             rand_nodes = self.draw_random_peers(nodes, num_select-len(selected), oracle)            

        self.state = selected + rand_nodes
        exploits = selected + rand_nodes
        explore_nodes = self.depleting_pool.get_exploring_peers(nodes, exploits, self.num_rand, oracle)
        self.state = selected + rand_nodes + explore_nodes

        return exploits, explore_nodes


    # nodes is a list whose value is node id
    def old_run_selector(self, H_in, nodes, unkn_plus_mask, plus_mask, unkn_unab_mask, oracle, curr_out):
        non_value_mask = 1*((plus_mask+unkn_unab_mask+unkn_plus_mask)>0)
        H = H_in.copy()*(1-non_value_mask) + 1e10*non_value_mask

        formatter.printt('\tExploit vs. Explore\n', self.log)
        num_select = self.out_lim-self.num_rand

        rand_nodes = []
        selected = self.subset_exploiter.select_subset_peer(H, nodes, num_select, plus_mask, oracle)
        if len(selected) != num_select:
            rand_nodes = self.draw_random_peers(nodes, num_select-len(selected), oracle)            

        self.state = selected + rand_nodes

        # selected, rand_nodes = self.get_best_ranks(H, nodes, self.out_lim-self.num_rand)
        # selected, rand_nodes = self.select_best_peer(H, nodes, self.out_lim-self.num_rand, plus_mask)
        # selected, rand_nodes = self.select_subset_peer(H, nodes, self.out_lim-self.num_rand, plus_mask, oracle)

        # select node to explore
        explore_nodes = self.depleting_pool.get_exploring_peers(nodes, selected+rand_nodes, self.num_rand)
        # explore_nodes = self.greedy_explorer.get_exploring_peers(H, nodes, plus_mask, selected+rand_nodes,self.num_rand)

        self.state = selected + rand_nodes + explore_nodes
        return selected + rand_nodes, explore_nodes
        

