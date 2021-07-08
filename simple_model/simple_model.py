import torch
import numpy as np
import sys
import config
import random

from collections import defaultdict
from net_init import load_network
from net_init import generate_random_outs_conns as gen_rand_outs
from network.sparse_table import SparseTable
from network.communicator import Communicator
from network import comm_network
from simple_model import formatter
from mat_complete.mat_comp_solver import construct_table
# from simple_model import mc_optimizer
from simple_model import mc_optimizer
from simple_model.selector import SimpleSelector
from utils.logger import Logger

torch.manual_seed(12)
np.random.seed(12)
random.seed(12)

class NetworkSim:
    def __init__(self, topo, num_out, num_in, num_epoch, T, num_topo, interested_i, mc_epochs, mc_lr, num_rand):
        self.T = T
        self.N = num_out
        self.H_ref = None
        self.loc = None
        self.ld = None
        self.proc_delay = None
        self.roles = None
        self.loc, self.ld, self.roles, self.proc_delay = load_network(topo)

        self.num_out = num_out
        self.num_in = num_in
        self.num_node = len(self.loc)
        self.num_epoch = num_epoch
        self.num_topo = num_topo
        self.num_rand = num_rand

        assert(T%num_topo==0)
        self.table_directions = ['incoming', 'outgoing', 'bidirect']

        self.nodes = {i: Communicator(i, self.proc_delay[i], num_in, num_out, []) 
                for i in range(self.num_node)}
        self.out_hist = []

        self.pubs = [k for k,v in self.roles.items() if v=='PUB']
        self.pub_hist = []
        self.sparse_tables = {i: SparseTable(i) for i in range(self.num_node)}

        self.star_i = interested_i
        self.logger = Logger('./', self.star_i, False)
        self.mco = mc_optimizer.McOptimizer(self.star_i, mc_epochs, mc_lr)
        self.selector = SimpleSelector(self.star_i, self.num_node,num_out,num_in, None, num_rand, self.logger)

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

    def choose_x_random(self, outs, x, star_i):
        keeps = np.random.choice(outs, x, replace=False)
        pools = [i for i in range(self.num_node) if i!=star_i and i not in outs]
        rands = np.random.choice(pools, len(outs)-x, replace=False)
        assert(len(set(keeps).intersection(set(rands))) == 0)
        return list(keeps) + list(rands)

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

    def has_missing_entry(self, table):
        num_entry = table.shape[0] * table.shape[1]
        return np.sum(table) != num_entry


    def run_mc(self, i ,curr_out):
        abs_slots = self.sparse_tables[i].abs_table[-self.T:]
        slots = self.sparse_tables[i].table[-self.T:]

        # debug abs table
        incomplete_table,M,nM,max_time,ids,ids_direct=construct_table(abs_slots, i, self.table_directions)
        topo_directions = formatter.get_topo_direction(slots, self.table_directions, self.num_topo)

        formatter.print_mats([
            formatter.format_double_masked_mat(incomplete_table, M, nM, topo_directions, self.num_topo, True, False),
            formatter.format_topo_array(self.pub_hist[-self.T:], self.num_topo, 'Pub'),
            ], 
        self.num_topo)


        incomplete_table,M,nM,max_time,ids,ids_direct=construct_table(slots, i, self.table_directions)
        topo_directions = formatter.get_topo_direction(slots, self.table_directions, self.num_topo)


        if self.has_missing_entry(M):
            completed_table, C = self.mco.run(incomplete_table, M, 1-nM, max_time)

            formatter.print_mats([
                formatter.format_double_masked_mat(incomplete_table, M, nM, topo_directions, self.num_topo, True,False),
                formatter.format_topo_array(self.pub_hist[-self.T:], self.num_topo, 'Pub'),
                # formatter.format_mat(completed_table, topo_directions, self.num_topo, True, False),
                formatter.format_topo_array(C, self.num_topo, '_C_'),
                formatter.format_mat(completed_table-C, topo_directions, self.num_topo, True, False),
                ], 
                self.num_topo)
        else:
            completed_table = incomplete_table
            formatter.print_mats([
                formatter.format_double_masked_mat(incomplete_table, M, nM, topo_directions, self.num_topo, True, False),
                formatter.format_topo_array(self.pub_hist[-self.T:], self.num_topo, 'Pub'),
                ],
                self.num_topo
            )

        # selected_outs = self.choose_x_random(curr_out, 1, i)
        selected_outs = self.selector.run_selector(completed_table, ids)
        return selected_outs

    def run(self):
        # setup conn network
        curr_outs = gen_rand_outs(self.num_out, self.num_in, self.num_node, 'n')
        self.selector.set_state(curr_outs[self.star_i])
        self.out_hist.append(curr_outs)
        self.setup_conn_graph(curr_outs)
        
        num_msg = int(self.T/self.num_topo)

        for e in range(self.num_epoch):
            ps = self.broadcast_msgs(num_msg)
            # for i in range(self.num_node):
                # print(i, self.nodes[i].outs, self.nodes[i].ins)
            # formatter.print_curr_conns(curr_outs, self.ld)
            # for i in range(self.num_node):
                # print('node', i, self.nodes[i].outs, self.nodes[i].ins, self.sparse_tables[i].abs_table)
            if e+1 >= self.num_topo:
                curr_outs[self.star_i] = self.run_mc(self.star_i, curr_outs[self.star_i])
            else:
                curr_outs[self.star_i] = self.choose_x_random(curr_outs[self.star_i], 1, self.star_i)
            self.setup_conn_graph(curr_outs)
            print('epoch', e, 'node', self.star_i, 'outs', curr_outs[self.star_i])


            
    # def construct_data(self, std, H_ref):
        # W = torch.zeros((T, L))
        # for i in range(T):
            # j = np.random.randint(L)
            # W[i,j] = 1.0
        # X = torch.matmul(W, H_ref)

        # X_noised = X + torch.randn(size=(T, N)) * std
        # X_noised[X_noised<0] = 0
        # return X, X_noised, W

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

