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

    # def get_ranks(self, H, node_ids):
        # # TODO it is possibel two column for a row has the same lowest entry, ignore now
        # row_mins = np.argmin(H, axis=1) 
        # ranks = defaultdict(int)
        # for r in row_mins:
            # index = node_ids[r]
            # ranks[index] += 1
        # return ranks

    # def get_ranks_with_plus(self, H, node_ids, plus_mask):
        # row_mins = np.argmin(H, axis=1) 
        # rank_plus = {} # key is peer, value is scores
        # for i, r in enumerate(row_mins):
            # index = node_ids[r]
            # if index not in rank_plus:
                # rank_plus[index] = RankPlus(index, 1, 0)
            # else:
                # rank_plus[index].rank += 1

            # num_plus = np.sum(plus_mask[i])
            # rank_plus[index].num_plus += num_plus
        # return rank_plus

    # def get_subset_ranks_with_plus(self, H, node_ids, plus_mask):
        # row_mins = np.argmin(H, axis=1) 
        # contributor_record = {} # key is peer, value is record
        # for i, r in enumerate(row_mins):
            # index = node_ids[r]
            # if index not in contributor_record:
                # contributor_record[index] = RankPlus(index, 1, 0)
            # else:
                # contributor_record[index].rank += 1

            # contributor_record[index].update_plus(plus_mask[i], node_ids)

        # for c, record in contributor_record.items():
            # formatter.printt('\t\t\tco:{} pp:{} rank:{}\n'.format(c, record.plus_cover, record.rank), self.log)

        # return contributor_record

    # currently use highest min
    # def get_best_ranks(self, H, nodes, num_select):
        # T = H.shape[0]
        # N = H.shape[1]       
        # ranks = self.get_ranks(H, nodes) 

        # sorted_ranks = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        
        # formatter.printt('\tExploit vs. Explore\n', self.log)
        # formatter.printt('\t\tranks by num min {}\n'.format(str(sorted_ranks)))
        # selected = []
        # if len(sorted_ranks) >= num_select:
            # for i in range(num_select):
                # node_id = sorted_ranks[i][0]
                # selected.append(node_id)
            # print('\t\tAdapt:\t\tselected '+str(selected))
            # return selected, []
        # else:
            # for i in range(len(sorted_ranks)):
                # node_id = sorted_ranks[i][0]
                # selected.append(node_id)

            # pools = set([i for i in range(self.num_node)])
            # pools = pools.difference(set(nodes))
            # pools.remove(self.id)
            # if len(pools) >= num_select-len(selected):
                # conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
                # print('\t\tExploit:Miss\t\t'+str(sorted(selected))+' rand '+str(conns))
            # else:
                # pools = set([i for i in range(self.num_node)])
                # pools = pools.difference(set(selected))
                # pools.remove(self.id)
                # conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
                # print('\t\tExploit:Miss+random\t\tselected '+str(sorted(selected))+' rand '+str(sorted(conns)))
            # return selected, conns

    # def sort_subset_rank_plus(self, contributor_rankplus, node_ids, num_select): 
        # subset_coverage_size = {}
        # subset_coverage_rank = {}
        # all_cands = sorted(contributor_rankplus.keys())
        # for comb in itertools.combinations(all_cands, num_select):
            # peer_contributors = defaultdict(dict) # key is benefiting peer, value is contributing peers 
            # coverage_rank = 0
            # for con in comb:
                # cs = contributor_rankplus[con]
                # for plus_cover, occurance in cs.plus_cover.items():
                    # for b in plus_cover:
                        # peer_contributors[b][con] = occurance
                        # # not the following, does not encourage many high coverage, may due to close 
                        # # coverage_rank += occurance
                    # coverage_rank += occurance


            # subset_coverage_size[comb] = len(peer_contributors) # simplest by coverage
            # subset_coverage_rank[comb] = coverage_rank
            # formatter.printt('\t\t\tsubset_score {} {} {} {}\n'.format(comb, subset_coverage_size[comb], subset_coverage_rank[comb], list(peer_contributors.keys())), self.log)

        # subset_tuple = []
        # for subset, coverage_size in subset_coverage_size.items():
            # total_rank = 0
            # coverage_rank = subset_coverage_rank[subset] 
            # for p in subset:
                # total_rank += contributor_rankplus[p].rank
            # subset_tuple.append((subset, total_rank, coverage_rank, coverage_size))
        # subset_tuple = sorted(subset_tuple, key=lambda tup: tup[1], reverse=True)
        # subset_tuple = sorted(subset_tuple, key=lambda tup: tup[2], reverse=True)
        # subset_tuple = sorted(subset_tuple, key=lambda tup: tup[3], reverse=True)
        # return subset_tuple

    # def sort_peers_by_count(self, H, nodes, plus_mask):
        # rank_plus = self.get_ranks_with_plus(H, nodes, plus_mask) 
        # rank_plus_tuple = [(n, int(rp.rank), int(rp.num_plus)) for n, rp in rank_plus.items() ]
        # # stable sort by rank
        # sorted_rank_plus = sorted(rank_plus_tuple, key=lambda tup: tup[1], reverse=True)
        # # then stable sort by plus
        # sorted_rank_plus = sorted(sorted_rank_plus, key=lambda tup: tup[2], reverse=True)
        # return sorted_rank_plus

    # def select_subset_peer(self, H, nodes, num_select, plus_mask, oracle):
        # T, N = H.shape
        # contributor_records = self.get_subset_ranks_with_plus(H, nodes, plus_mask)
        # all_cands = sorted(contributor_records.keys())
        # if len(all_cands) >= num_select:
            # sorted_subset = self.sort_subset_rank_plus(contributor_records, nodes, num_select)
            # selected = list(sorted_subset[0][0])
            # formatter.printt('\t\tranks by num plus cover {}\n'.format(sorted_subset), self.log)
            # formatter.printt('\t\tExploit(subset):\t\t{}\n'.format(selected), self.log)
            # return selected, []
        # else:
            # selected = all_cands
            # conns = self.draw_random_peers(nodes, num_select-len(selected))
            # formatter.printt('\t\tExploit(insuff):\t\t{}\n'.format(selected), self.log)
            # return selected, conns
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
            # for i in pools:
                # if len(oracle.can_i_connect(self.id, [i])) == 0:
                    # conns.append(i)
                # if num == len(conns):
                    # break
            # if num != len(conns):
                # print('cannot find ', num, 'peers satisfying oracle')
                # sys.exit(1)

            formatter.printt('\t\tExploit(random):\t\t{}\n'.format(conns), self.log)
        else:
            pools = set([i for i in range(self.num_node)])
            pools = pools.difference(set(selected))
            pools.remove(self.id)
            np.random.shuffle(pools)
            # conns = list(np.random.choice(list(pools), num_select-len(selected), replace=False))
            conns = self.find_x_rand_to_conn(pools, num, oracle)

            # for i in pools:
                # if len(oracle.can_i_connect(self.id, [i])) == 0:
                    # conns.append(i)
                # if num == len(conns):
                    # break
            # if num != len(conns):
                # print('cannot find ', num, 'peers satisfying oracle')
                # sys.exit(1)

            formatter.printt('\t\tExploit(random+reuse):\t\t{}\n'.format(conns), self.log)
        return conns

    def set_state(self, outs):
        self.state = outs.copy()

    def run_selector(self, H_in, nodes, unkn_plus_mask, plus_mask, unkn_unab_mask, oracle, curr_out):
        non_value_mask = 1*((plus_mask+unkn_unab_mask+unkn_plus_mask)>0)
        H = H_in.copy()*(1-non_value_mask) + 1e10*non_value_mask
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
        

