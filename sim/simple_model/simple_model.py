import torch
import numpy as np
import sys
import config
import random
import math
import pickle
import json
import os
import time

from multiprocessing.pool import Pool
from collections import defaultdict
from net_init import load_network
from net_init import generate_random_outs_conns as gen_rand_outs
from net_init import generate_random_outs_conns_with_oracle as gen_rand_outs_with_oracle
from network.sparse_table import SparseTable
from network.communicator import Communicator
from network import comm_network
from network.oracle import SimpleOracle
from simple_model import formatter
from simple_model.scheduler import NodeScheduler
from mat_complete.mat_comp_solver import construct_table
# from simple_model import mc_optimizer
from simple_model import mc_optimizer
from simple_model.selector import SimpleSelector
from utils.logger import Logger
import networkx as nx



# pick your fav seed

def delineate_out_actions(old_outs, new_outs):
    rm_outs =  [i for i in old_outs if i not in new_outs]
    add_outs = [i for i in new_outs if i not in old_outs]
    return rm_outs, add_outs

class NetworkSim:
    def __init__(self, topo, num_out, num_in, num_epoch, T, num_topo, stars, mc_epochs, mc_lr, mc_exit_loss_diff, num_rand, top_n_peer, plt_name, print_log, churn_rate, update_interval):
        self.T = T
        self.N = num_out
        self.H_ref = None
        self.loc = None
        self.ld = None
        self.proc_delay = None
        self.roles = None
        self.loc, self.ld, self.roles, self.proc_delay, self.pub_prob = load_network(topo)
    
        s = 0
        for k,v in self.pub_prob.items():
            # if self.roles[k] == 'PUB':
            s += v
        print('publishing prob summation', s)

        self.num_out = num_out
        self.num_in = num_in
        self.num_node = len(self.loc)
        self.num_epoch = num_epoch
        self.num_topo = num_topo
        self.num_rand = num_rand
        self.update_interval = update_interval

        self.churn_rate = churn_rate
        
        assert(T%num_topo==0)
        self.table_directions = ['incoming', 'outgoing', 'bidirect']

        self.nodes = {i: Communicator(i, self.proc_delay[i], num_in, num_out, []) 
                for i in range(self.num_node)}
        self.oracle = SimpleOracle(num_in, num_out, self.num_node)
        self.out_hist = []

        # self.pubs = [k for k,v in self.roles.items() if v=='PUB']



        self.pub_hist = []
        self.sparse_tables = {i: SparseTable(i) for i in range(self.num_node)}

        self.stars = stars 
        self.schedulers = {i: NodeScheduler(i, self.update_interval, self.num_topo) for i in self.stars}

        # self.mco = mc_optimizer.McOptimizer(self.star_i, mc_epochs, mc_lr, mc_exit_loss_diff, top_n_peer)
        dirpath = os.path.dirname(plt_name)
        if not print_log:
            self.log_files = {i: os.path.join(dirpath, 'log'+str(i)+'.txt') for i in self.stars} 
        else:
            self.log_files = {i: None for i in self.stars}

        self.mcos = {i: mc_optimizer.McOptimizer(i, mc_epochs, mc_lr, mc_exit_loss_diff, top_n_peer, self.log_files[i])
                for i in self.stars}

        # self.selector = SimpleSelector(self.star_i, self.num_node,num_out,num_in, None, num_rand)
        self.selectors = {i: SimpleSelector(i, self.num_node, num_out, num_in, None, num_rand, self.log_files[i])
                for i in self.stars}
        
        self.dist_file = plt_name
 
        self.init_graph_conn = os.path.join(dirpath, 'init.json')
        self.dists_hist = defaultdict(list)
        self.snapshot_dir = os.path.join(dirpath, 'snapshots')
        self.snapshot_exploit_dir = os.path.join(dirpath, 'snapshots-exploit')
        self.write_adapts_node(os.path.join(dirpath, 'adapts'))

        self.epoch_graph_dir = os.path.join(dirpath, 'graphs')

        self.num_thread = min(64, len(self.stars))
        self.worker_pools = Pool(processes=self.num_thread) 

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

    def get_exploring_peers(self, curr_peers, outs, star_i):
        explorer = self.selectors[star_i].explorer
        peers = explorer.get_exploring_peers(curr_peers, outs[:self.num_out-self.num_rand], self.num_rand, self.oracle)
        return peers

    def random_explore(self, outs, star_i):
        keeps = outs[:self.num_out-self.num_rand]
        pools = [i for i in range(self.num_node) if i!=star_i and i not in outs]
        np.random.shuffle(pools)
        rands = []
        for i in pools:
            un = self.oracle.can_i_connect(star_i, [i])
            if len(un) == 0:
                rands.append(i)
            if self.num_rand == len(rands):
                break
        if self.num_rand != len(rands):
            print('cannot find ', len(outs)-self.num_rand, 'peers satisfying oracle')
            sys.exit(1)
        assert(len(set(keeps).intersection(set(rands))) == 0)
        rm_outs, add_outs = delineate_out_actions(outs, keeps+rands)
        self.oracle.update(star_i, [], add_outs, rm_outs)
        return list(keeps) + list(rands)

    def choose_x_random(self, outs, x, star_i):
        keeps = list(np.random.choice(outs, len(outs)-x, replace=False))
        pools = [i for i in range(self.num_node) if i!=star_i and i not in outs]
        np.random.shuffle(pools)
        rands = []
        for i in pools:
            un = self.oracle.can_i_connect(star_i, [i])
            if len(un) == 0:
                rands.append(i)
            if x == len(rands):
                break

        if x != len(rands):
            print('cannot find ', x, 'peers satisfying oracle')
            sys.exit(1)

        assert(len(set(keeps).intersection(set(rands))) == 0)

        rm_outs, add_outs = delineate_out_actions(outs, keeps+rands)
        # print(star_i, 'outs', outs, 'keep', keeps,'rand', rands)
        # print('rm_outs', rm_outs, 'add_outs', add_outs)
        self.oracle.update(star_i, [], add_outs, rm_outs)
        return list(keeps) + list(rands)

    def broadcast_msgs(self, num_msg):
        time_tables = {i:defaultdict(list) for i in range(self.num_node)}
        abs_time_tables = {i:defaultdict(list) for i in range(self.num_node)}
        broads = []

        pubs = []
        probs = []
        for k, v in self.pub_prob.items():
            pubs.append(k)
            probs.append(v)

        for _ in range(num_msg):
            p = np.random.choice(pubs, size=1, replace=False, p=probs)[0]
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

    def has_missing_entry(self, table):
        num_entry = table.shape[0] * table.shape[1]
        return np.sum(table) != num_entry

        
    def run_mc(self, i, curr_out, e):
        slots = self.sparse_tables[i].table[-self.T:]
        incomplete_table,M,nM,max_time,ids,ids_direct = construct_table(slots, i, self.table_directions)
        topo_directions = formatter.get_topo_direction(slots, self.table_directions, self.num_topo)

        if self.has_missing_entry(M):
            completed_table, C, unkn_plus_mask, unkn_unab_mask = self.mcos[i].run(incomplete_table, M, 1-nM, max_time)
            formatter.print_mats([
                formatter.format_double_masked_mat(incomplete_table, M, nM, topo_directions, self.num_topo, True,False),
                formatter.format_topo_array(self.pub_hist[-self.T:], self.num_topo, 'Pub'),
                # formatter.format_mat(completed_table, topo_directions, self.num_topo, True, False),
                formatter.format_topo_array(C, self.num_topo, '_C_'),
                formatter.format_completed_mat(completed_table, unkn_unab_mask, unkn_plus_mask, nM, topo_directions, self.num_topo, True,False),
                # formatter.format_mat(completed_table, topo_directions, self.num_topo, True, False),
                ], 
                self.num_topo, 
                self.log_files[i])
        else:
            completed_table = incomplete_table
            unkn_plus_mask = np.zeros(completed_table.shape)
            unkn_unab_mask = np.zeros(completed_table.shape)
            formatter.print_mats([
                formatter.format_double_masked_mat(incomplete_table, M, nM, topo_directions, self.num_topo, True, False),
                formatter.format_topo_array(self.pub_hist[-self.T:], self.num_topo, 'Pub'),
                ],
                self.num_topo,
                self.log_files[i]
            )

        exploits, explores = self.selectors[i].run_selector(completed_table, ids, unkn_plus_mask, nM, unkn_unab_mask, self.oracle, curr_out)

        rm_outs, add_outs = delineate_out_actions(curr_out, exploits+explores)
        self.oracle.update(i, [], add_outs, rm_outs)

        # self.get_truth_distance(i, exploits, e)
        return exploits + explores

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

    def write_epoch_graph(self, e, curr_outs):
        filename = os.path.join(self.epoch_graph_dir, 'epoch'+str(e)+'.json')
        with open(filename, 'w') as w:
            graph_json = []
            for u in range(self.num_node):
                node = self.nodes[u]
                # outs = sorted([int(i) for i in node.outs])
                num_exploit = self.num_out - self.num_rand
                exploits = curr_outs[u][:num_exploit]
                explores = curr_outs[u][num_exploit:]
                assert(len(node.outs.difference(curr_outs[u])) == 0)
                ins = sorted([int(i) for i in node.ins])
                peer = {
                    'node': int(u),
                    'exploits': [int(i) for i in exploits],
                    'explores': [int(i) for i in explores],
                    'ins': ins
                    }
                graph_json.append(peer)
            json.dump(graph_json, w, indent=4)

    def write_adapts_node(self, filename):
        with open(filename, 'w') as w:
            sorted_stars = sorted(self.stars)
            for star in sorted_stars:
                w.write(str(star) + '\n')

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
            if i in self.stars:
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
        formatter.printt('\tEval peers {}\n'.format(interested_peers), self.log_files[star_i])

        pubs = [k for k,v in self.roles.items() if v=='PUB']
        for m in pubs:
            # the closest distance
            start_t = time.time()
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
                line_len = self.ld[star_i][m] + self.proc_delay[m]
                # line_len_comp =  int(math.ceil(math.sqrt(
                    # (self.loc[star_i][0]-self.loc[m][0])**2+
                    # (self.loc[star_i][1]-self.loc[m][1])**2 ))) +self.proc_delay[m]
                # assert(line_len_comp == line_len)

            dists[m] = (j, round(topo_length, 3), round(line_len, 3))
            runtime = round(time.time() - start_t)
            dist_text = "\t\tpub {m} by peer {j} opt-diff {opt_diff} topo_len {topo_len} line_len {line_len} in {runtime} sec\n".format(m=m,j=j,opt_diff=round(topo_length-line_len,1),topo_len=round(topo_length),line_len=round(line_len),runtime=runtime)
            formatter.printt(dist_text, self.log_files[star_i])
        self.dists_hist[star_i].append((epoch, dists))

    def log_epoch(self, e, curr_outs, churn_stars):
        topo_text = ""
        for star_i in self.stars:
            star_i_text = "\t\t {} outs {}\n".format(star_i, curr_outs[star_i])
            topo_text += star_i_text

        for star_i in self.stars:
            epoch_text = "\n\n\t\t\t*** Epoch {e} stars {stars}  ***\n".format(e=e,stars=self.stars)
            epoch_text = "\n\n\t\t\t*** Epoch {e} churn stars {stars}  ***\n".format(e=e,stars=churn_stars)
            epoch_text += topo_text   
            epoch_text += '\n'
            formatter.printt(epoch_text, self.log_files[star_i])

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

    def take_snapshot(self, epoch, curr_outs):
        name = "epoch"+str(epoch)+".txt"
        outpath = os.path.join(self.snapshot_dir, name)
        self.write_cost(outpath)
        outpath_exploit = os.path.join(self.snapshot_exploit_dir, name)
        self.write_exploit_cost(outpath_exploit, curr_outs)

    def run(self):
        # setup conn network
        # curr_outs = gen_rand_outs(self.num_out, self.num_in, self.num_node, 'n')
        curr_outs = gen_rand_outs_with_oracle(self.num_out, self.num_node, self.oracle)
        self.oracle.check(curr_outs)

        for star_i, selector in self.selectors.items():
            selector.set_state(curr_outs[star_i])
        self.out_hist.append(curr_outs)
        self.setup_conn_graph(curr_outs)

        # self.oracle.setup(curr_outs)
        self.write_init_graph()
        
        num_msg = int(self.T/self.num_topo)
        churn_stars = []

        for e in range(self.num_epoch):
            self.log_epoch(e, curr_outs, churn_stars)
            self.take_snapshot(e, curr_outs)
            self.write_epoch_graph(e, curr_outs)
            self.oracle.check(curr_outs)
            ps = self.broadcast_msgs(num_msg)

            churn_stars = comm_network.get_network_churning_nodes(self.churn_rate, self.stars)
            for star_i in churn_stars:
                self.schedulers[star_i].initiate_update(e)
            mc_stars, acc_stars = self.run_scheduler(e)

            if len(mc_stars) > 0:
                if self.num_thread == 1:
                    # single thread
                    for star_i in mc_stars:
                        curr_outs[star_i] = self.run_mc(star_i, curr_outs[star_i], e)
                else:
                    # parallel threads
                    outputs = self.run_parallel_mc(curr_outs, e, mc_stars)
                    for star_i, exploit_explore in outputs.items():
                        curr_outs[star_i] = exploit_explore[0] + exploit_explore[1]

            if len(acc_stars) > 0:
                for star_i in acc_stars:
                    curr_peers = list(self.nodes[star_i].get_peers())
                    last_out = curr_outs[star_i]
                    rands = self.get_exploring_peers(curr_peers, last_out, star_i) 
                    curr_outs[star_i] = last_out[:self.num_out-self.num_rand] + rands
                    self.oracle.update(star_i, [], rands, last_out[-self.num_rand:])

            for star_i in self.stars:
                self.get_truth_distance(star_i, curr_outs[star_i][:self.num_out-self.num_rand], e)

            self.setup_conn_graph(curr_outs)
            
            
        self.save_dists_hist()

    def run_scheduler(self, e):
        mc_stars = []
        acc_stars = []
        for star_i in self.stars:
            state = self.schedulers[star_i].get_curr_state(e)
            if state == 'run':
                mc_stars.append(star_i)
            elif state == 'acc':
                acc_stars.append(star_i)
        return mc_stars, acc_stars

    def run_parallel_mc(self, curr_outs, e, churn_stars):
        stars_inputs = {}
        for i in churn_stars:
            slots = self.sparse_tables[i].table[-self.T:]
            incomplete_table,M,nM,max_time,ids,ids_direct=construct_table(slots, i, self.table_directions)
            topo_directions = formatter.get_topo_direction(slots, self.table_directions, self.num_topo)
            stars_inputs[i] = [
                    i,curr_outs[i],incomplete_table,M,nM,max_time,ids,ids_direct,
                    topo_directions, self.log_files[i], self.mcos[i], self.selectors[i]]

        stars = np.random.permutation(churn_stars)
        outputs = run_parallel_mc_(self.worker_pools, stars, stars_inputs, self.num_topo, self.pub_hist, self.T, self.oracle)
        for star_i in churn_stars:
            exploits = outputs[star_i][0]
            # self.get_truth_distance(star_i, exploits, e)
        return outputs

def mc_run_(mco, incomplete_table, M, nM, max_time, star_i):
   completed_table, C, unkn_plus_mask, unkn_unab_mask = mco.run(incomplete_table, M, 1-nM, max_time)
   return completed_table, C, unkn_plus_mask, unkn_unab_mask, star_i 
    

def run_parallel_mc_(pools, stars, stars_input, num_topo, pub_hist, T, oracle):
    args = []
    for star_i in stars:
        i,curr_out,incomplete_table,M,nM,max_time,ids,ids_direct,topo_directions,log_file,mco,selector=stars_input[star_i]
        arg = (mco, incomplete_table, M, nM, max_time, star_i)
        args.append(arg)

    results = pools.starmap(mc_run_, args)
    assert(len(results) == len(stars))
    outputs = {}
    for j in range(len(stars)):
        completed_table, C, unkn_plus_mask, unkn_unab_mask, star_i = results[j]
        r_i,curr_out,incomplete_table,M,nM,max_time,ids,ids_direct,topo_directions,log_file,mco,selector=stars_input[star_i]

        formatter.print_mats([
                formatter.format_double_masked_mat(incomplete_table, M, nM, topo_directions, num_topo, True,False),
                formatter.format_topo_array(pub_hist[-T:], num_topo, 'Pub'),
                formatter.format_topo_array(C, num_topo, '_C_'),
                formatter.format_completed_mat(completed_table, unkn_unab_mask, unkn_plus_mask, nM, topo_directions, num_topo, True,False),
                ], 
                num_topo, 
                log_file)

        exploits, explores = selector.run_selector(completed_table, ids, unkn_plus_mask, nM, unkn_unab_mask, oracle, curr_out)
        
        rm_outs, add_outs = delineate_out_actions(curr_out, exploits+explores)
        oracle.update(star_i, [], add_outs, rm_outs)
        outputs[star_i] = [exploits, explores]

    return outputs


# # # config # # #
# T = 30
# N = 5 
# L = 2
# num_mask_per_row = 1 #int(N/3)
# mean = 20
# std =  5
# offset = 10
# epochs = 2000
# row_penalty = 10 
# lr = 1e-3
# topo = '../topo/dc2-5node-1pub-no-proc.json'

# # # exp setup # # #
# H_lat = torch.rand(L, N) * 2*mean + offset
# X, X_noised, W = construct_data(T, L, std, H_lat)
# region_labels = torch.argmax(W, axis=1).numpy()
# M = torch.ones((T, N))
# for i in range(T):
    # masks = np.random.permutation(N)[:num_mask_per_row]
    # M[i,masks] = 0

# X_answer = X_noised.numpy()
# X_answer = X_answer - np.reshape(np.min(X_answer, axis=1), (T,1))

# X_m = X_noised * M
# min_times = torch.zeros((T,1))
# for i in range(T):
    # min_e = None
    # for j in range(N):
        # e = X_m[i,j]
        # if (min_e is None or min_e > e) and M[i,j].item()>0:
            # min_e = e
    # min_times[i,0] = min_e

# # # observed # #
# X_rel = M*(X_noised - min_times)

# logger.print_mats([logger.format_masked_mat(X_rel, M.numpy(), True), 
    # logger.format_array(region_labels, 'label'), 
    # logger.format_mat(M, True)])


# H = torch.rand(T, N) # * torch.max(X_rel) 
# C = torch.rand(T, 1, requires_grad=True) 
# H.requires_grad_()

# criterion = CompletionLoss()
# prox_plus = torch.nn.Threshold(0,0)

# for e in range(epochs):
    # X_input = X_rel 
    # loss = criterion(X_input, H, C, M)
    # loss.backward()

    # with torch.no_grad():
        # s = (X-(H-C) )*M
        # if e % 100 ==0:
            # print(e, 'normalized loss', torch.norm(s))

        # H = prox_plus(H - lr * H.grad)
        # C = prox_plus(C - lr * C.grad)

        # H.requires_grad_()
        # C.requires_grad_()

        # H.grad = None
        # C.grad = None

# # # # # Eval # # #
# with torch.no_grad():
    # print('X_noised, X_masked, X_rel')
    # logger.print_mats([
        # logger.format_mat(X_noised.numpy(), True), 
        # logger.format_array(region_labels, 'label'), 
        # logger.format_masked_mat(X_noised.numpy(),M.numpy(), True), 
        # logger.format_masked_mat(X_rel.numpy(), M.numpy(), True)
        # ])
    # s = (X-(H-C) )*M

    # print('normalized loss', torch.norm(s))
    # a = np.round(H.numpy(), 2)
    # num = [i for i in range(T)]
    # print('H \t\t C')
    # logger.print_mats([
        # logger.format_array(num, 'n'), 
        # logger.format_mat(a, True), 
        # logger.format_mat(C.numpy(), True), 
        # logger.format_array(region_labels, 'label'),
        # logger.format_mat(X_answer, True), 
        # logger.format_masked_mat(X_rel.numpy(), M.numpy(), True), 
        # logger.format_mat(H.numpy()-X_answer, True)
        # ])
    # print(np.linalg.norm(H.numpy()-X_answer))

# for i in range(self.num_node):
            # print(i, curr_outs[i])
        # curr_ins = self.get_curr_ins(curr_outs)
        # max_in = 0
        # for i in range(self.num_node):
            # max_in = max(max_in, len(curr_ins[i]))
            # print(i, curr_ins[i])
        # print(max_in)
