import numpy as np
from collections import defaultdict
import formatter
import sys

class DepletingPool:
    def __init__(self, i, known_peers, log):
        self.hist_explored_peers = {} # key is peer, value is the last time in mc table
        self.counter = 0
        self.id = i
        self.known_peers = set(known_peers)
        self.log = log
        self.dead_loop_breaker = 0

    def get_exploring_peers(self, curr_peers, keep_peers, num_explore, oracle, curr_out):
        self.dead_loop_breaker += 1
        if self.dead_loop_breaker > 2:
            print('called get_exploring_peers more than 2. Cannot explore new peers even in a whole pool. Choose ranom non-outgoin peer to explore', self.dead_loop_breaker)
            cands = self.known_peers.difference(curr_out)
            explores = list(np.random.choice(list(cands), num_explore, replace=False))
            return [int(i) for i in  explores]

            # sys.exit(1)
        for p in curr_peers + keep_peers:
            self.hist_explored_peers[p] = self.counter
        pools = self.known_peers
        cands = list(pools.difference(self.hist_explored_peers.keys()))
        # print(self.counter, 'seen', len(self.hist_explored_peers), ' peers', sorted(list(self.hist_explored_peers.keys())))
        if len(cands) >= num_explore:
            explores = []
            np.random.shuffle(cands)
            for i in cands:
                if len(oracle.can_i_connect(self.id, [i])) == 0:
                    explores.append(i)
                if num_explore == len(explores):
                    break

            if num_explore == len(explores):
                # explores = list(np.random.choice(list(cands), num_explore, replace=False))
                formatter.printt('\t\tExplore(deplet full):\t\t{}\n'.format(sorted(explores)), self.log)
                self.counter += 1
                self.dead_loop_breaker = 0
                return explores
            else:
                num_explore -= len(explores)
                self.hist_explored_peers.clear()
                new_pool_explore = self.get_exploring_peers(curr_peers, keep_peers, num_explore, oracle, curr_out)
                formatter.printt('\t\tExplore(deplet insu oracle):\t\t{}\n'.format(sorted(explores+new_pool_explore)),self.log)
                self.counter += 1
                self.dead_loop_breaker = 0
                return explores + new_pool_explore

        else:
            explores = []
            for i in cands:
                if len(oracle.can_i_connect(self.id, [i])) == 0:
                    explores.append(i)
                if num_explore == len(explores):
                    break

            num_explore -= len(cands)
            self.hist_explored_peers.clear()
            new_pool_explore = self.get_exploring_peers(curr_peers, keep_peers, num_explore, oracle, curr_out)
            # print('\t\tExplore(deplet):\t\t'+str(sorted(explores+new_pool_explore)))
            formatter.printt('\t\tExplore(deplet insu cand):\t\t{}\n'.format(sorted(explores+new_pool_explore)),self.log)
            self.counter += 1
            self.dead_loop_breaker = 0
            return explores + new_pool_explore

class RandomExplorer:
    def __init__(self, i, known_peers, log):
        self.id = i
        self.known_peers = set(known_peers)
        self.log = log

    def get_exploring_peers(self, curr_peers, keep_peers, num_explore, oracle):
        pools = self.known_peers
        cands = list(pools.difference(curr_peers))
        explores = []
        for i in np.random.permutation(cands):
            if len(oracle.can_i_connect(self.id, [i])) == 0:
                explores.append(i)
            if num_explore == len(explores):
                break
        if num_explore != len(explores):
            print('cannot find sufficient random nodes')
            sys.exit(1)
        return explores


class GreedyExplorer:
    def __init__(self, node_id):
        self.subset_record = defaultdict(dict) # key is subset, value is covered peered thourgh me whose value is count
        self.peer_record = {} # contributor who gives good record
        self.id = node_id

    def select_comparison(self):
        pass

    def get_exploring_peers(self, H, nodes, plus_mask, curr_peers, keep_peers, num_explore):
        pass
