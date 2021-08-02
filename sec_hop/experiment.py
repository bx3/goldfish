import sys
import pickle
import json
import os
import math
import networkx as nx
from collections import defaultdict

from net_init import load_network
from net_init import generate_random_outs_conns_with_oracle as gen_rand_outs_with_oracle
from network.sparse_table import SparseTable
from network.communicator import Communicator
from network import comm_network
from network.oracle import SimpleOracle
from sec_hop.selector import Selector
from mat_complete.mat_comp_solver import construct_table
import random
import numpy as np


class Experiment:
    def __init__(self, topo, in_lim, out_lim, name, num_keep, num_2hop, num_rand, num_epoch, adapts, num_msg, churn_rate):
        self.in_lim = in_lim
        self.out_lim = out_lim
        self.num_out = out_lim
        self.outdir = os.path.dirname(name)

        self.loc, self.ld, self.roles, self.proc_delay = load_network(topo)
        self.num_node = len(self.loc) 
        self.num_epoch = num_epoch
        self.num_msg = num_msg
        self.churn_rate = churn_rate

        self.selectors = {i: Selector(i, num_keep, num_rand, num_msg, self.num_node)
                for i in range(self.num_node)} 
        # elf.num_cand = num_node # could be less if num_node is large

        self.snapshots = []
        # self.pools = Pool(processes=num_thread) 

        self.directions = ['incoming', 'outgoing', 'bidirect']
        self.nodes = {i: Communicator(i, self.proc_delay[i], in_lim, out_lim, []) 
                for i in range(self.num_node)}
        self.oracle = SimpleOracle(in_lim, out_lim, self.num_node)
        self.out_hist = []
        self.sparse_tables = {i: SparseTable(i) for i in range(self.num_node)}

        # self.conns_snapshot = []
        # self.broad_nodes = [] # hist of broadcasting node
        # self.timer = time.time()

        self.pubs = [k for k,v in self.roles.items() if v=='PUB']
        self.adapts = adapts
        self.pub_hist = []
        self.dists_hist = defaultdict(list)
        self.dist_file = name 

        # log setting
        # self.use_logger = use_logger 
        # self.logdir = self.outdir + '/' + 'logs'
        # if not os.path.exists(self.logdir):
            # os.makedirs(self.logdir)
        # self.loggers = {}
        # self.init_logger()

        self.init_graph_conn = os.path.join(self.outdir, 'init.json')
        self.snapshot_dir = os.path.join(self.outdir, 'snapshots')
        self.snapshot_exploit_dir = os.path.join(self.outdir, 'snapshots-exploit')
        self.write_adapts_node(os.path.join(self.outdir, 'adapts'))

        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)


        self.num_keep = num_keep
        self.num_2hop = num_2hop
        self.num_rand = num_rand
        assert(num_keep + num_2hop + num_rand == self.out_lim)

    def construct_graph(self):
        G = nx.Graph()
        for i, node in self.nodes.items():
            for u in node.outs:
                delay = self.ld[i][u] + node.node_delay/2 + self.nodes[u].node_delay/2
                if i == u:
                    print('self loop', i)
                    sys.exit(1)
                G.add_edge(i, u, weight=delay)
        return G

    def construct_exploit_graph(self, curr_outs):
        G = nx.Graph()
        for i, node in self.nodes.items():
            out_peers = []
            if i in self.adapts:
                out_peers = curr_outs[i][:self.num_out-self.num_rand]
            else:
                out_peers = curr_outs[i]

            for u in out_peers:
                delay = self.ld[i][u] + node.node_delay/2 + self.nodes[u].node_delay/2
                if i == u:
                    print('self loop', i)
                    sys.exit(1)
                G.add_edge(i, u, weight=delay)
        return G

    def write_cost(self, outpath):
        G = self.construct_graph()
        with open(outpath, 'w') as w:
            length = dict(nx.all_pairs_dijkstra_path_length(G))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    cost = length[i][j] - self.proc_delay[i]/2.0 + self.proc_delay[j]/2.0
                    w.write(str(cost) + ' ')
                w.write('\n')

    def write_exploit_cost(self, outpath, curr_outs):
        G = self.construct_exploit_graph(curr_outs)
        with open(outpath, 'w') as w:
            length = dict(nx.all_pairs_dijkstra_path_length(G))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    cost = length[i][j] - self.proc_delay[i]/2.0 + self.proc_delay[j]/2.0
                    w.write(str(cost) + ' ')
                w.write('\n')

    def write_adapts_node(self, filename):
        with open(filename, 'w') as w:
            sorted_stars = sorted(self.adapts)
            for star in sorted_stars:
                w.write(str(star) + '\n')

    def get_truth_distance(self, star_i, interested_peers, epoch):
        # construct graph
        G = nx.Graph()
        for i, node in self.nodes.items():
            if i == star_i:
                for u in interested_peers:
                    # only connect interested edge from the interested node
                    delay = self.ld[i][u] + node.node_delay/2 + self.nodes[u].node_delay/2
                    if i == u:
                        print('self loop', i)
                        sys.exit(1)
                    G.add_edge(i, u, weight=delay)
            else:
                for u in node.outs:
                    # not connecting incoming edge to the interested node
                    if u != star_i:
                        delay = self.ld[i][u] + node.node_delay/2 + self.nodes[u].node_delay/2
                        if i == u:
                            print('self loop', i)
                            sys.exit(1)
                        G.add_edge(i, u, weight=delay)
        dists = {} # key is the target pub, value is the best peer and length
        for m in self.pubs:
            # the closest distance
            length, path = nx.single_source_dijkstra(G, source=star_i, target=m, weight='weight')
            assert(len(path)>=0) 
            topo_length = None 
            line_len = None
            j = None
            if len(path) == 1:
                # itself
                assert(star_i == m)
                topo_length = 0
                line_len = 0
                j = star_i
            else:
                j = path[1]
                topo_length = length - self.proc_delay[j]/2.0 + self.proc_delay[m]/2.0
                line_len = (math.sqrt(
                    (self.loc[star_i][0]-self.loc[m][0])**2+
                    (self.loc[star_i][1]-self.loc[m][1])**2 ) +self.proc_delay[m])

            dists[m] = (j, round(topo_length, 3), round(line_len, 3))
        self.dists_hist[star_i].append((epoch, dists))

    def save_dists_hist(self):
        if self.dist_file == 'None':
            return
        with open(self.dist_file, 'wb') as w:
            pickle.dump(self.dists_hist, w)

    def write_init_graph(self):
        with open(self.init_graph_conn, 'w') as w:
            graph_json = []
            for u in range(self.num_node):
                node = self.nodes[u]
                outs = sorted([int(i) for i in node.outs])
                ins = sorted([int(i) for i in node.ins])
                peer = {
                    'node': int(u),
                    'outs': outs,
                    'ins': ins
                    }
                graph_json.append(peer)
            json.dump(graph_json, w, indent=4)


    def take_snapshot(self, epoch, curr_outs):
        name = "epoch"+str(epoch)+".txt"
        outpath = os.path.join(self.snapshot_dir, name)
        self.write_cost(outpath)
        outpath_exploit = os.path.join(self.snapshot_exploit_dir, name)
        self.write_exploit_cost(outpath_exploit, curr_outs)


    # def init_selectors(self, out_conns, in_conns):
        # for u in range(self.num_node):
            # # if smaller then it is adv
            # if u in self.adversary.sybils:
                # self.selectors[u] = Selector(u, True, out_conns[u], in_conns[u], None)
            # else:
                # self.selectors[u] = Selector(u, False, out_conns[u], in_conns[u], None)

    def broadcast_msgs(self, num_msg):
        time_tables = {i:defaultdict(list) for i in range(self.num_node)}
        abs_time_tables = {i:defaultdict(list) for i in range(self.num_node)}
        broads = []
        for _ in range(num_msg):
            p = random.choice(self.pubs)
            self.pub_hist.append(p)
            broads.append(p)

            comm_network.broadcast_msg(
                p, 
                self.nodes, 
                self.ld, 
                time_tables, 
                abs_time_tables
                )
        for i in range(self.num_node):
            self.sparse_tables[i].append_time(abs_time_tables[i], num_msg, 'abs_time')
            self.sparse_tables[i].append_time(time_tables[i], num_msg, 'rel_time')

        return broads

    def update_selectors(self, outs_conns, ins_conn):
        for i in range(self.num_node):
            self.selectors[i].update(outs_conns[i], ins_conn[i])

    def get_curr_ins(self, curr_outs):
        curr_ins = defaultdict(list)
        for u in range(self.num_node):
            for o in curr_outs[u]:
                curr_ins[o].append(u)
        return curr_ins

    def setup_conn_graph(self, curr_outs):
        curr_ins = self.get_curr_ins(curr_outs)
        for u in range(self.num_node):
            self.nodes[u].update_conns(curr_outs[u], curr_ins[u])


    def run_2hop(self, adapt_i, curr_out, e): 
        slots = self.sparse_tables[adapt_i].table[-self.num_msg:]
        incomplete_table,M,nM,max_time,ids,ids_direct = construct_table(slots, adapt_i, self.directions)
        selected, rands = self.selectors[adapt_i].run(self.oracle, curr_out, ids, slots)

        return selected + rands

    def run(self):
        curr_outs = gen_rand_outs_with_oracle(self.num_out, self.num_node, self.oracle)
        self.oracle.check(curr_outs)

        self.setup_conn_graph(curr_outs)
        self.write_init_graph()

        for e in range(self.num_epoch):
            self.take_snapshot(e, curr_outs)
            self.oracle.check(curr_outs)
            ps = self.broadcast_msgs(self.num_msg)
            churn_adapts = comm_network.get_network_churning_nodes(self.churn_rate, self.adapts)
            for adapt_i in np.random.permutation(churn_adapts):
                curr_outs[adapt_i] = self.run_2hop(adapt_i, curr_outs[adapt_i], e)
            self.setup_conn_graph(curr_outs)
            for adapt_i in self.adapts:
                self.get_truth_distance(adapt_i, curr_outs[adapt_i][:self.num_keep], e)

        self.save_dists_hist()





        # while True:
            # network_state.reset(self.num_node, self.in_lim)

            # if num_snapshot == len(record_epochs):
                # break

            # if self.method == 'mc':
                # outs_conns, start_mc = self.run_mc(max_epoch,record_epochs, num_msg, epoch, network_state)
                # self.conns_snapshot.append(outs_conns)

                # if epoch in record_epochs:
                    # self.take_snapshot(epoch)
                    # num_snapshot += 1

            # elif self.method == '2hop':
                # outs_conns = self.run_2hop(num_msg, epoch, network_state)
                # self.conns_snapshot.append(outs_conns)

                # if epoch in record_epochs:
                    # self.take_snapshot(epoch)
                    # num_snapshot += 1

            # epoch += 1

    # def select_nodes(nodes, ld, num_msg, selectors, oracle, update_nodes, time_tables, in_lim, out_lim, network_state, num_keep, num_2hop, num_random):
        # outs_neighbors = {} # output container
        # num_invalid_compose = 0
        # # direct peers
        # num_rand_1hop = 0
        # for i in update_nodes:
            # keep_candidates = list(nodes[i].outs | nodes[i].ins )

            # composes = comb_subset.get_config(
                    # num_keep, 
                    # keep_candidates,
                    # len(keep_candidates), 
                    # network_state,
                    # i)
            
            # num_invalid_compose += math.comb(len(keep_candidates), num_keep) - len(composes)
            # if len(composes) == 0:
                # peers = selectors[i].select_random_peers(nodes, num_keep, network_state)
                # num_rand_1hop += 1
                # # oracle needs to know the connection
                # oracle.update_1_hop_peers(i, peers)
                # outs_neighbors[i] = peers
            # else:
                # for compose in composes:
                    # if len(compose) != len(set(compose)):
                        # print('repeat in compose')
                        # print(i)
                        # print('composes', compose)
                        # print(keep_candidates)
                        # print('in', list(nodes[i].outs))
                        # print('out', list(nodes[i].ins))
                        # sys.exit(1)

                # peers = selectors[i].select_1hops(time_tables[i], composes, num_msg, network_state)
                # # oracle needs to know the connection
                # oracle.update_1_hop_peers(i, peers)
                # outs_neighbors[i] = peers


        # num_added_2hop = 0
        # num_added_3hop = 0
        # num_added_random = 0
        # tot_not_seen = 0
        # random.shuffle(update_nodes)
        # # two hop peers
        # if num_2hop > 0:
            # for u in update_nodes:
                # peers_info = oracle.get_multi_hop_info(u)
                # peers, num_not_seen = selectors[u].select_peers(
                        # config.num_2_hop, nodes, peers_info.two_hops, network_state)
                # oracle.update_2_hop_peers(u, peers)
                # outs_neighbors[u] += peers
                # num_added_2hop += len(peers)

                # tot_not_seen += num_not_seen
                
                # # add 3hops
                # if out_lim - len(outs_neighbors[u]) > num_random:
                    # num_3_hop = out_lim - len(outs_neighbors[u]) - num_random
                    # peers_info = oracle.get_multi_hop_info(u)
                    # peers, num_not_seen = selectors[u].select_peers(num_3_hop, nodes, peers_info.three_hops, network_state)
                    # oracle.update_3_hop_peers(u, peers)
                    # outs_neighbors[u] += peers
                    # num_added_3hop += len(peers) 
                    # tot_not_seen += num_not_seen
        
        # # add random
        # for u in update_nodes:
            # num_random = out_lim - len(outs_neighbors[u]) 
            # num_added_random += num_random

            # peers = selectors[u].select_random_peers(nodes, num_random, network_state)
            # for p in peers:
                # if p in outs_neighbors[u]:
                    # print(p, 'in neigbors', outs_neighbors[u])
                    # sys.exit(1)
            # outs_neighbors[u] += peers

        # # debug
        # for u in update_nodes:
            # if len(set(outs_neighbors[u])) != out_lim:
                # print(u, "has less out neighbors")
                # print(outs_neighbors[u])
                # print(selectors[u].desc_conn)
                # sys.exit(1)
        # print('num_rand_1hop', num_rand_1hop,'num_invalid_compose', num_invalid_compose )
        # # print('Finish. num2hop', num_added_2hop, 'num3hop', num_added_3hop, 'num rand', num_added_random, 'num no seen', tot_not_seen)
        # return outs_neighbors

