import numpy as np
from collections import defaultdict
from collections import namedtuple
from mat_complete.mat_comp_solver import print_mats
from mat_complete.mat_comp_solver import format_mat
from mat_complete.mat_comp_solver import format_array
from simple_model import explorer
from simple_model import formatter
import random
import sys
import itertools


class RankPlus:
    def __init__(self, peer, rank, plus):
        self.peer = peer
        self.rank = int(rank) # occurance of argmin for all rows
        self.num_plus = int(plus)
        self.plus_cover = {} # key is indicator rows with plus being 1, this pubs may be good for two sources, value is count

    def update_plus(self, row_plus, node_ids):
        plus_benefited_peers = []
        for i in range(len(row_plus)):
            if row_plus[i] != 0:
                plus_benefited_peers.append(node_ids[i])
        # maybe unnecessary, but make sure sorted

        if len(plus_benefited_peers) > 0:
            plus_benefited_peers = tuple(sorted(plus_benefited_peers))
            if plus_benefited_peers not in self.plus_cover:
                self.plus_cover[plus_benefited_peers] = 1
            else:
                self.plus_cover[plus_benefited_peers] += 1
        
        self.num_plus += np.sum(row_plus)


class SimpleSelector:
    def __init__(self, i, num_node, out_lim, in_lim, state, num_rand, logfile):
        self.id = i
        self.num_node = num_node
        self.known_peers = [j for j in range(self.num_node) if i != j]
        self.out_lim = out_lim
        self.in_lim = in_lim
        self.best_count = 1
        self.state = state # states is the out conns
        self.num_rand = num_rand
        self.temperature = 1.5
        self.depleting_pool = explorer.DepletingPool(self.known_peers, logfile)
        self.greedy_explorer = explorer.GreedyExplorer(i)
        self.log = logfile


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

    def get_ranks_with_plus(self, H, node_ids, plus_mask):
        row_mins = np.argmin(H, axis=1) 
        rank_plus = {} # key is peer, value is scores
        for i, r in enumerate(row_mins):
            index = node_ids[r]
            if index not in rank_plus:
                rank_plus[index] = RankPlus(index, 1, 0)
            else:
                rank_plus[index].rank += 1

            num_plus = np.sum(plus_mask[i])
            rank_plus[index].num_plus += num_plus
        return rank_plus

    def get_subset_ranks_with_plus(self, H, node_ids, plus_mask):
        row_mins = np.argmin(H, axis=1) 
        contributor_record = {} # key is peer, value is record
        for i, r in enumerate(row_mins):
            index = node_ids[r]
            if index not in contributor_record:
                contributor_record[index] = RankPlus(index, 1, 0)
            else:
                contributor_record[index].rank += 1

            contributor_record[index].update_plus(plus_mask[i], node_ids)

        for c, record in contributor_record.items():
            formatter.printt('\t\t\tco:{} pp:{} rank:{}\n'.format(c, record.plus_cover, record.rank), self.log)

        return contributor_record

    # currently use highest min
    def get_best_ranks(self, H, nodes, num_select):
        T = H.shape[0]
        N = H.shape[1]       
        ranks = self.get_ranks(H, nodes) 

        sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        
        formatter.printt('\tExploit vs. Explore\n', self.log)
        formatter.printt('\t\tranks by num min {}\n'.format(str(sorted_ranks)))
        selected = []
        if len(sorted_ranks) >= num_select:
            for i in range(num_select):
                node_id = sorted_ranks[i][0]
                selected.append(node_id)
            print('\t\tAdapt:\t\tselected '+str(selected))
            return selected, []
        else:
            for i in range(len(sorted_ranks)):
                node_id = sorted_ranks[i][0]
                selected.append(node_id)

            pools = set([i for i in range(self.num_node)])
            pools = pools.difference(set(nodes))
            pools.remove(self.id)
            if len(pools) >= num_select-len(selected):
                conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
                print('\t\tExploit:Miss\t\t'+str(sorted(selected))+' rand '+str(conns))
            else:
                pools = set([i for i in range(self.num_node)])
                pools = pools.difference(set(selected))
                pools.remove(self.id)
                conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
                print('\t\tExploit:Miss+random\t\tselected '+str(sorted(selected))+' rand '+str(sorted(conns)))
            return selected, conns

    def sort_subset_rank_plus(self, contributor_rankplus, node_ids, num_select): 
        subset_coverage_size = {}
        subset_coverage_rank = {}
        all_cands = sorted(contributor_rankplus.keys())
        for comb in itertools.combinations(all_cands, num_select):
            peer_contributors = defaultdict(dict) # key is benefiting peer, value is contributing peers 
            coverage_rank = 0
            for con in comb:
                cs = contributor_rankplus[con]
                for plus_cover, occurance in cs.plus_cover.items():
                    for b in plus_cover:
                        peer_contributors[b][con] = occurance
                        # not the following, does not encourage many high coverage, may due to close 
                        # coverage_rank += occurance
                    coverage_rank += occurance


            subset_coverage_size[comb] = len(peer_contributors) # simplest by coverage
            subset_coverage_rank[comb] = coverage_rank
            formatter.printt('\t\t\tsubset_score {} {} {} {}\n'.format(comb, subset_coverage_size[comb], subset_coverage_rank[comb], list(peer_contributors.keys())), self.log)

        subset_tuple = []
        for subset, coverage_size in subset_coverage_size.items():
            total_rank = 0
            coverage_rank = subset_coverage_rank[subset] 
            for p in subset:
                total_rank += contributor_rankplus[p].rank
            subset_tuple.append((subset, total_rank, coverage_rank, coverage_size))
        subset_tuple = sorted(subset_tuple, key=lambda tup: tup[1], reverse=True)
        subset_tuple = sorted(subset_tuple, key=lambda tup: tup[2], reverse=True)
        subset_tuple = sorted(subset_tuple, key=lambda tup: tup[3], reverse=True)
        return subset_tuple

    def sort_peers_by_count(self, H, nodes, plus_mask):
        rank_plus = self.get_ranks_with_plus(H, nodes, plus_mask) 
        rank_plus_tuple = [(n, int(rp.rank), int(rp.num_plus)) for n, rp in rank_plus.items() ]
        # stable sort by rank
        sorted_rank_plus = sorted(rank_plus_tuple, key=lambda tup: tup[1], reverse=True)
        # then stable sort by plus
        sorted_rank_plus = sorted(sorted_rank_plus, key=lambda tup: tup[2], reverse=True)
        return sorted_rank_plus

    def draw_random_peers(self, excludes, num):
        pools = set([i for i in range(self.num_node)])
        pools = pools.difference(set(excludes))
        pools.remove(self.id)
        conns = None
        if len(pools) >= num:
            conns = list(np.random.choice(list(pools), num, replace=False))
            # print('\t\tExploit:Rand\t\t'+str(conns))
            formatter.printt('\t\tExploit(random):\t\t{}\n'.format(conns), self.log)
        else:
            pools = set([i for i in range(self.num_node)])
            pools = pools.difference(set(selected))
            pools.remove(self.id)
            conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
            # print('\t\tExploit:Rand+Reuse\t\t'+str(sorted(conns)))
            formatter.printt('\t\tExploit(random+reuse):\t\t{}\n'.format(conns), self.log)
        return conns

    def select_best_peer(self, H, nodes, num_select, plus_mask):
        T = H.shape[0]
        N = H.shape[1]       
        sorted_rank_plus = self.sort_peers_by_count(H, nodes, plus_mask)
        formatter.printt('\t\tranks by num plus {}\n'.format(sorted_rank_plus), self.log)

        # TODO it is possible two col have a double tie, maybe add some random number
        selected = []
        if len(sorted_rank_plus) >= num_select:
            for i in range(num_select):
                node_id = sorted_rank_plus[i][0]
                selected.append(node_id)
            formatter.printt('\t\tExploit( plus ):\t\t{}\n'.format(selected), self.log)
            return selected, []
        else:
            for i in range(len(sorted_rank_plus)):
                node_id = sorted_rank_plus[i][0]
                selected.append(node_id)
            # print('\t\tExploit:Insufficient\t\t'+str(selected))
            formatter.printt('\t\tExploit(insuff):\t\t{}\n'.format(selected), self.log)
            conns = self.draw_random_peers(nodes, num_select-len(selected))
            return selected, conns

    def select_applicable_subset(self, oracle, sorted_subset):
        selected = [] 
        for subset in sorted_subset:
            comb = subset[0]
            unconnectable = oracle.can_i_connect(self.id, comb)
            if len(unconnectable) == 0:
                selected = comb
                break
        return selected


    def select_subset_peer(self, H, nodes, num_select, plus_mask, oracle):
        T, N = H.shape
        contributor_records = self.get_subset_ranks_with_plus(H, nodes, plus_mask)
        all_cands = sorted(contributor_records.keys())
        if len(all_cands) >= num_select:
            sorted_subset = self.sort_subset_rank_plus(contributor_records, nodes, num_select)
            selected = list(sorted_subset[0][0])
            formatter.printt('\t\tranks by num plus cover {}\n'.format(sorted_subset), self.log)
            formatter.printt('\t\tExploit(subset):\t\t{}\n'.format(selected), self.log)
            return selected, []
        else:
            selected = all_cands
            conns = self.draw_random_peers(nodes, num_select-len(selected))
            formatter.printt('\t\tExploit(insuff):\t\t{}\n'.format(selected), self.log)
            return selected, conns

        
    def set_state(self, outs):
        self.state = outs.copy()

    # nodes is a list whose value is node id
    def run_selector(self, H_in, nodes, unkn_plus_mask, plus_mask, unkn_unab_mask, oracle):
        # select node to keep
        non_value_mask = 1*((plus_mask+unkn_unab_mask+unkn_plus_mask)>0)
        H = H_in.copy()*(1-non_value_mask) + 1e10*non_value_mask

        formatter.printt('\tExploit vs. Explore\n', self.log)
        # selected, rand_nodes = self.get_best_ranks(H, nodes, self.out_lim-self.num_rand)
        # selected, rand_nodes = self.select_best_peer(H, nodes, self.out_lim-self.num_rand, plus_mask)
        selected, rand_nodes = self.select_subset_peer(H, nodes, self.out_lim-self.num_rand, plus_mask, oracle)

        # select node to explore
        self.state = selected + rand_nodes

        explore_nodes = self.depleting_pool.get_exploring_peers(nodes, selected+rand_nodes, self.num_rand)
        # explore_nodes = self.greedy_explorer.get_exploring_peers(H, nodes, plus_mask, selected+rand_nodes,self.num_rand)

        self.state = selected + rand_nodes + explore_nodes
        return selected + rand_nodes, explore_nodes
        

