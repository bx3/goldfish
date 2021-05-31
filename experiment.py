import sys
import os
import config
from oracle import PeersInfo
from communicator import Note
import networkx as nx
import writefiles
import schedule 
import adversary
from oracle import NetworkOracle 
from selector import Selector
from collections import defaultdict
from schedule import NetworkState
import time
import numpy as np
import comm_network
import random
from optimizer import Optimizer
from optimizer import SparseTable
from bandit import Bandit
import initnetwork
import math
import logger
import solver
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, node_hash, link_delay, node_delay, num_node, in_lim, out_lim, num_region, name, sybils, window, method, loc):
        self.nh = node_hash
        self.ld = link_delay
        self.nd = node_delay
        self.loc = loc
        self.num_node = num_node
        self.nodes = {} # nodes are used for communicating msg in network
        self.conns = {} # key is node, value if number of connection
        self.selectors = {} # selectors for choosing outgoing conn for next time
        self.in_lim = in_lim
        self.out_lim = out_lim
        self.num_region = num_region
        self.outdir = name

        self.timer = time.time()

        self.num_cand = num_node # could be less if num_node is large

        self.adversary = adversary.Adversary(sybils)
        self.snapshots = []

        self.pools = Pool(processes=config.num_thread) 

        self.optimizers = {}
        self.sparse_tables = {}
        self.bandits = {}
        self.window = window 

        self.init_conns = None

        self.broad_nodes = [] # hist of broadcasting node
        self.method = method
        self.batch_type = 'rolling' # or rolling
        self.broad_method = 'fixed_miners' # fixed_point, rand, hash_dist
        self.topo_type = 'rand'
        self.fixed_miners = None

        # init dc parameter
        if self.topo_type == 'dc':
            self.nodes_per_region = int((self.num_node-1) / self.num_region)

            self.init_conns = initnetwork.gen_dc_connected(
                self.num_node,
                config.num_dc_region,
                self.out_lim,
                'y'
                )
        elif self.topo_type == 'rand':
            self.init_conns = initnetwork.generate_random_outs_conns(
                self.out_lim,
                3, # TODO self.in_lim,
                self.num_node,
                'y'
                )
        else:
            print('Unknown topo type')
            sys.exit(1)

        self.save_conn_plot() 


        if self.broad_method == 'fixed_miners':
            if self.topo_type == 'dc':
                nodes_per_region = int((self.num_node - 1)/(self.num_region))
                self.fixed_miners = [i*nodes_per_region+1 for i in range(self.num_region)]
            elif self.topo_type == 'rand':
                self.fixed_miners = [15,24,30]
                self.get_dist_among_fixed_points(self.init_conns.copy(), self.out_lim, self.fixed_miners)

                # self.fixed_miners = self.get_rand_graph_fixed_points(self.init_conns.copy(), self.out_lim)

        # log setting
        self.printer = Printer(num_node, out_lim, 'log/WH_hist', 'log/ucb')
        self.logdir = self.outdir + '/' + 'logs'
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.loggers = {}
        self.init_logger()
        self.log0 = self.outdir + '/' + '0.log'
        with open(self.log0, 'w') as w:
            pass

    def save_conn_plot(self):
        conns = self.init_conns.copy()
        conns.pop(0, None)
        print('init conns', conns)
        G = nx.Graph()
        for i, cs in conns.items():
            for j in cs:
                G.add_edge(i, j) #, lat = self.ld[i,j]
        nx.draw(G, pos=self.loc, with_labels=True, node_size=96) # 
        # edge_labels = nx.get_edge_attributes(G,'lat')
        # nx.draw_networkx_edge_labels(G, self.loc)
        plt.savefig(self.outdir + '/' + 'topo.png')


    def get_dist_among_fixed_points(self, outs_conns, num_choose, fixed_miners):
        outs_conns.pop(0, None)
        G = initnetwork.construct_graph_by_outs_conns(outs_conns, self.nd, self.ld)
        num = len(fixed_miners)
        for i in range(num):
            m = fixed_miners[i]
            for j in range(i+1, num):
                n = fixed_miners[j]
                distance, path = nx.single_source_dijkstra(G, m, target=n)
                print('dist',m,'->',n, distance, path)


    # three points
    def get_rand_graph_fixed_points(self, outs_conns, num_choose):
        outs_conns.pop(0, None)
        G = initnetwork.construct_graph_by_outs_conns(outs_conns, self.nd, self.ld)
        miners = []
        miner_cands = [i for i in range(1, len(outs_conns))]

        u = np.random.choice(miner_cands)
        miners.append(u)
        miner_cands.remove(u)
        length, paths = nx.single_source_dijkstra(G, u)
        sorted_len = sorted(length.items(), key=lambda x: x[1])

        sorted_path = sorted(paths, key=lambda k: len(paths[k]))
        print(paths)
        print(sorted_path)
        furthest_point = sorted_path[-1]
        furthest_path = paths[furthest_point]

        miners.append(furthest_point)

        space = int(len(sorted_len)/(num_choose-1))
        for u in reversed(range(len(sorted_path)-1)):
            i = sorted_path[u]
            print(i)
            path = set(paths[i])
            if path.intersection(set(furthest_path)) != len(path):
                miners.append(i)

            if len(miners) == num_choose:
                break
        print('fixed miners are', miners)
        miners_len = [length[i] for i in miners]

        length, paths = nx.single_source_dijkstra(G, miners[-1])
        print('distance are', miners_len)
        print('distance are', length[miners[1]])

        return miners


    # generate networkx graph instance
    def construct_graph(self):
        G = nx.Graph()
        for i, node in self.nodes.items():
            for u in node.outs:
                delay = self.ld[i][u] + node.node_delay/2 + self.nodes[u].node_delay/2
                assert(i != u)
                G.add_edge(i, u, weight=delay)
        return G




    def update_ins_conns(self):
        all_nodes = list(self.nodes.keys())
        conns_ins = defaultdict(list)
        for i in all_nodes:
            node = self.nodes[i]
            for out in node.outs:
                self.nodes[out].ins.add(i)
                conns_ins[out].append(i)
        return conns_ins

    def update_conns(self, out_conns):
        # correctness check
        num_double_conn = 0

        #for i, peers in out_conns.items():
        for i in range(len(out_conns)):
            peers = out_conns[i]
            for p in peers:
                if i in out_conns[p]:
                    # print(i, out_conns[i])
                    # print(p, out_conns[p])
                    # print('')
                    num_double_conn += 1
        if num_double_conn> 0:
            pass
            # print("num_double_conn > 0", num_double_conn)

        nodes = self.nodes
        for i in range(len(out_conns)):
            peers = out_conns[i]
            if len(set(peers)) != len(peers):
                print('Error. Repeat peer')
                print(i, peers)
                sys.exit(1)
            nodes[i].ordered_outs = peers.copy()
            nodes[i].outs = set(peers)
            nodes[i].ins.clear()
            nodes[i].recv_time = 0
            nodes[i].received = False
        conn_ins = self.update_ins_conns()
        return conn_ins

    def get_outs_neighbors(self):
        out_neighbor = np.zeros((self.num_node, self.out_lim))
        for i in range(self.num_node):
            out_neighbor[i] = self.nodes[i].ordered_outs
        return out_neighbor

    def init_optimizers(self):
        for i in range(self.num_node):
            self.optimizers[i] = Optimizer(
                i,
                self.num_cand,
                self.num_region,
                self.window,
                self.batch_type
            )
    def init_sparse_tables(self):
        for i in range(self.num_node):
            self.sparse_tables[i] = SparseTable(
                i,
                self.num_node,
                self.num_region,
                self.window,
            )

    def init_bandits(self):
        for i in range(self.num_node):
            self.bandits[i] = Bandit(
                i,
                self.num_region,
                self.num_node,
                config.alpha,
                config.ucb_method
            )

    def init_logger(self):
        for i in range(self.num_node):
            self.loggers[i] = logger.Logger(self.logdir, i)
            self.loggers[i].write_miners(self.fixed_miners)

    def init_graph(self):
        outs_conns = self.init_conns.copy()
        for i in range(self.num_node):
            node_delay = self.nd[i]
            self.nodes[i] = Note(
                i,
                node_delay,
                self.in_lim,
                self.out_lim,
                outs_conns[i]
            )
        ins_conns = self.update_ins_conns()
        self.init_selectors(outs_conns, ins_conns)
        self.init_optimizers()
        self.init_sparse_tables()
        self.init_bandits()

        # for i in range(config.num_node):
            # print(i, outs_neighbors[i], ins_conns[i])
        # sys.exit(1)

    def take_snapshot(self, epoch):
        name =  str(config.network_type)+'_'+str(config.method)+"V1"+"Round"+str(epoch)+".txt"
        outpath = self.outdir + "/" + name
            
        G = self.construct_graph()
        outs_neighbors = self.get_outs_neighbors()

        # structure_name =  self.outdir + "/" + 'structure_' +  str(epoch) + '.txt'
        # writefiles.write_conn(structure_name, outs_neighbors)

        writefiles.write(outpath, G, self.nd, outs_neighbors, self.num_node)

        curr_time = time.time()
        elapsed = curr_time - self.timer 
        self.timer = curr_time
        print(" * Recorded at the end of ", epoch)

    def init_selectors(self, out_conns, in_conns):
        for u in range(self.num_node):
            # if smaller then it is adv
            if u in self.adversary.sybils:
                self.selectors[u] = Selector(u, True, out_conns[u], in_conns[u], None)
            else:
                self.selectors[u] = Selector(u, False, out_conns[u], in_conns[u], None)

    def broadcast_msgs(self, num_msg):
        # key is id, value is a dict of peers whose values are lists of relative timestamp
        time_tables = {i:defaultdict(list) for i in range(self.num_node)}
        abs_time_tables = {i:defaultdict(list) for i in range(self.num_node)}
        broad_nodes = []
        for _ in range(num_msg):
            broad_node = -1
            if self.broad_method == 'fixed_miners':
                broad_node = np.random.choice(self.fixed_miners)
            elif self.broad_method == 'rand':
                broad_node = np.random.choice([i for i in range(1, self.num_node)])
            elif self.broad_method == 'hash_dist':
                broad_node = comm_network.get_broadcast_node(self.nh)
            else:
                print('Unknown. broadcast method')
                sys.exit(1)
            broad_nodes.append(broad_node)
            self.broad_nodes.append(broad_node)
            comm_network.broadcast_msg(
                broad_node, 
                self.nodes, 
                self.ld, 
                self.nh, 
                time_tables, 
                abs_time_tables
                )
        return time_tables, abs_time_tables, broad_nodes

    def update_selectors(self, outs_conns, ins_conn):
        for i in range(self.num_node):
            self.selectors[i].update(outs_conns[i], ins_conn[i])
            
    def shuffle_nodes(self):
        update_nodes = None
        if not config.sybil_update_priority:
            update_nodes = [i for i in range(self.num_node)]
            random.shuffle(update_nodes)
        else:
            # make sure sybils node knows the information first
            all_nodes = set([i for i in range(self.num_node)])
            honest_nodes = list(all_nodes.difference(set(self.adversary.sybils)))
            random.shuffle(honest_nodes)
            update_nodes = sybils + honest_nodes
        assert(update_nodes != None and len(update_nodes) == self.num_node)
        return update_nodes

    def accumulate_optimizer(self, time_table, abs_time_tables, num_msg):
        for i in range(self.num_node):
            if config.use_abs_time:
                self.optimizers[i].append_time(abs_time_tables[i], num_msg)
            else:
                self.optimizers[i].append_time(time_table[i])

    def accumulate_table(self, time_table, abs_time_tables, num_msg, broad_nodes):
        for i in range(self.num_node):
            # broadcasting nodes cannot receive msg from others
            if i not in broad_nodes:
                if config.use_abs_time:
                    self.sparse_tables[i].append_time(abs_time_tables[i], num_msg)
                else:
                    self.sparse_tables[i].append_time(time_table[i])
    def decide_need_iter(self):
        for i in range(self.num_node):
            if len(self.sparse_tables[i].table) < self.window:
                return True 
        return False 

    def evaluate_conns_on_fixed_miners(self, node_order, outs_conns, fixed_miners, loggers):
        G = initnetwork.construct_graph_by_outs_conns(outs_conns, self.nd, self.ld)
        for u in node_order:
            logger = loggers[u]
            distances, paths = nx.single_source_dijkstra(G, u)
            print(u, outs_conns[u])
            min_scores = {}
            for m in fixed_miners:
                off = distances[m] - self.ld[u, m]
                print(paths[m])
                min_scores[m] = (m, paths[m][1], off)
            
            sorted_scores_by_miner = []
            for m in fixed_miners:
                sorted_scores_by_miner.append(min_scores[m])
            logger.write_score(sorted_scores_by_miner)
            
    def start_rel_comp(self, max_epoch, record_epochs, num_msg):
        network_state = NetworkState(self.num_node, self.in_lim) 
        outs_conns = self.init_conns.copy()
        prev_conns = outs_conns.copy()
        init_write_epoch = None
        while True:
            oracle = NetworkOracle(config.is_dynamic, self.adversary.sybils, self.selectors)
            network_state.reset(self.num_node, self.in_lim)
            need_iter = self.decide_need_iter()
            if not need_iter:
                print('\t\t < epoch', epoch, 'start>')
                if init_write_epoch is None:
                    init_write_epoch = epoch

                time_tables, abs_time_tables, broad_nodes = self.broadcast_msgs(num_msg)
                self.accumulate_table(time_tables, abs_time_tables, num_msg, broad_nodes)
                with open(self.log0, 'a') as w:
                    str_list = [str(x) for x in outs_conns[0]]
                    w.write(str(epoch) + '\t' + '\t'.join(str_list) + '\n')

                total_msg += num_msg

                # matrix factorization
                # node_order = self.shuffle_nodes()
                node_order = [0] #np.random.permutation([i for i in range(self.num_node)])[:3]

                outs_conns = schedule.run_mf(
                    self.nodes, 
                    self.ld, 
                    self.nd,
                    self.nh, 
                    self.optimizers,
                    self.sparse_tables,
                    self.bandits,
                    node_order, 
                    time_tables, 
                    abs_time_tables,
                    self.in_lim,
                    self.out_lim, 
                    self.num_region,
                    network_state,
                    num_msg,
                    self.pools,
                    self.loggers,
                    epoch,
                    prev_conns,
                    self.broad_nodes,
                    self.topo_type,
                    self.fixed_miners
                    )

                prev_conns = outs_conns.copy()


        print('start_rel_comp')


    def start(self, max_epoch, record_epochs, num_msg):
        network_state = NetworkState(self.num_node, self.in_lim) 
        total_msg = 0
        time_tables = None
        if not config.use_matrix_completion:
            self.window = 0
        outs_conns = self.init_conns.copy()
        prev_conns = outs_conns.copy()
        num_snapshot = 0
        init_write_epoch = None 
        epoch = 0
        while True:
            oracle = NetworkOracle(config.is_dynamic, self.adversary.sybils, self.selectors)
            network_state.reset(self.num_node, self.in_lim)

            if self.method == 'mf-bandit':
                # print(outs_conns)
                epoch_start = time.time()
                need_iter = self.decide_need_iter()
                if not need_iter:
                    print('\t\t < epoch', epoch, 'start>')
                    if init_write_epoch is None:
                        init_write_epoch = epoch

                    time_tables, abs_time_tables, broad_nodes = self.broadcast_msgs(num_msg)
                    self.accumulate_table(time_tables, abs_time_tables, num_msg, broad_nodes)
                    with open(self.log0, 'a') as w:
                        str_list = [str(x) for x in outs_conns[0]]
                        w.write(str(epoch) + '\t' + '\t'.join(str_list) + '\n')

                    total_msg += num_msg

                    # matrix factorization
                    # node_order = self.shuffle_nodes()
                    node_order = [0] #np.random.permutation([i for i in range(self.num_node)])[:3]

                    outs_conns = schedule.run_mf(
                        self.nodes, 
                        self.ld, 
                        self.nd,
                        self.nh, 
                        self.optimizers,
                        self.sparse_tables,
                        self.bandits,
                        node_order, 
                        time_tables, 
                        abs_time_tables,
                        self.in_lim,
                        self.out_lim, 
                        self.num_region,
                        network_state,
                        num_msg,
                        self.pools,
                        self.loggers,
                        epoch,
                        prev_conns,
                        self.broad_nodes,
                        self.topo_type,
                        self.fixed_miners
                        )

                    prev_conns = outs_conns.copy()
                    # structure_name =  self.outdir + "/" + 'structure_' +  str(epoch-init_write_epoch) + '.txt'
                    # writefiles.write_conn(structure_name, outs_conns)

                    print(outs_conns)
                    ins_conn = self.update_conns(outs_conns)

                    self.evaluate_conns_on_fixed_miners(
                        node_order,
                        outs_conns,
                        self.fixed_miners,
                        self.loggers
                        )

                    if epoch-init_write_epoch in record_epochs:
                        print(epoch-init_write_epoch)
                        self.take_snapshot(epoch-init_write_epoch)
                        num_snapshot += 1

                    print('\t\t [ epoch', epoch, 'end', round(time.time() - epoch_start, 2), ']')                                  

                    if num_snapshot == len(record_epochs):
                        print('exit')
                        print(num_snapshot, len(record_epochs))
                        break 
                else:
                    # random connections
                    time_tables, abs_time_tables, broad_nodes = self.broadcast_msgs(1)
                    self.accumulate_table(time_tables, abs_time_tables, 1, broad_nodes)
                    # self.log_table(epoch, 1)
                    
                    total_msg += 1
                    rand_peers = []
                    for _ in range(self.out_lim):
                        cands = [i for i in range(1, self.num_node)]
                        k = np.random.choice(cands) 
                        while 0 == k or k in rand_peers: 
                            k = np.random.choice(cands)         
                        rand_peers.append(k)

                    outs_conns[0] = rand_peers
                    # print('rand outs_conns')
                    # print(outs_conns)
                    # print(outs_conns)
                    # outs_conns = initnetwork.generate_cluster_outs_conns(
                        # self.out_lim, 
                        # self.in_lim, 
                        # self.num_node,
                        # self.num_region,
                        # 0,
                        # True,
                        # )
                    # print('clusetr')
                    # outs_conns = initnetwork.generate_random_outs_conns(
                        # self.out_lim, 
                        # self.in_lim, 
                        # self.num_node)
                    prev_conns = outs_conns.copy()
                # updates connections
                    ins_conn = self.update_conns(outs_conns)


            elif self.method == '2hop':
                # 1,2,3 hop selection
                if epoch in record_epochs:
                    self.take_snapshot(epoch)
                time_tables, abs_time_table, broad_nodes = self.broadcast_msgs(num_msg)
                # node_order = self.shuffle_nodes()
                node_order = [7]
                outs_conns = schedule.select_nodes(
                    self.nodes, 
                    self.ld, 
                    num_msg, 
                    self.nh, 
                    self.selectors,
                    oracle,
                    node_order, 
                    time_tables, 
                    self.in_lim,
                    self.out_lim, 
                    network_state
                    )
                # update outs ins
                ins_conn = self.update_conns(outs_conns)
                # self.check()
                self.update_selectors(outs_conns, ins_conn)
            else:
                print('Error. Unknown method', self.method)
                sys.exit(1)


            epoch += 1
            # print(epoch, len(self.selectors[0].seen), sorted(self.selectors[0].seen))
        # self.printer.print_WH()
        # self.printer.print_ucb()

    def check(self):
        for i in range(self.num_node):
            out_conns = self.selectors[i].desc_conn
            num = 0
            for u, v in out_conns:
                if u == i:
                    num += 1
            if num != 3:
                print(i, out_conns)
                sys.exit(1)

    def log_conns(self, epoch):
        for i in range(self.num_node):
            node = self.nodes[i]
            outs = node.ordered_outs
            ins = sorted(node.ins)

            # self.loggers[i].write_conns(outs, ins, '>' + str(epoch))

    def log_table(self, epoch, num_msg):
        for i in range(self.num_node):
            st = self.sparse_tables[i]
            X, mask, _ = solver.construct_table(st.N, st.table[-self.window:])
            # TODO temporary debug hack, assuming there are 4 nodes per regions datacenter
            sr = str(int(self.broad_nodes[-1]/4 == int(i/8)))
            comment = '>'+str(epoch)+'. time table. Xnode:'+str(self.broad_nodes[-num_msg:])+'.sr'+sr
            self.loggers[i].write_mat(X, comment)

    def start_complete_graph(self, max_epoch, record_epochs):
        start = time.time()
        for epoch in range(max_epoch):
            print('epoch', epoch)
            if epoch in record_epochs:
                self.take_snapshot(epoch)
        finish = time.time()
        print(finish - start, 'elapsed')

    # def analytical_complete_graph(self):
        # print('start analytical analyze')
        # name =  str(config.network_type)+'_'+str(config.method)+"V1"+"Round"+'0'+".txt"
        # outpath = self.outdir + "/" + name
        # with open(outpath, 'w') as w:
            # for i in range(self.num_node):
                # for j in range(self.num_node):
                    # if i == j:
                        # delay = 0
                    # else:
                        # delay = self.ld[i][j] + self.nd[j]
                    # w.write(str(delay) + '  ')
                # w.write('\n')

# used for debug
class Printer:
    def __init__(self, n, r, filename, ucb_filename):
        self.H_mean_list = defaultdict(list) 
        self.H_max_list = defaultdict(list)
        self.H_nz_list = defaultdict(list)

        self.W_mean_list = defaultdict(list)
        self.W_max_list = defaultdict(list)
        self.W_nz_list = defaultdict(list)
        self.opt_list = defaultdict(list)
        self.H_list = defaultdict(list)

        self.num_node = n
        self.num_region = r
        self.log_solver_filename = filename
        self.log_ucb_filename = ucb_filename

        self.bandits = defaultdict(list)

    def update_WH(self, i, W, H, opt):
        self.H_list[i].append(H.copy())
        self.H_mean_list[i].append(round(np.mean(H), 2))
        self.H_max_list[i].append(round(np.max(H), 2))
        self.H_nz_list[i].append(np.count_nonzero(H==0))
        self.W_mean_list[i].append(round(np.mean(W), 2))
        self.W_max_list[i].append(round(np.max(W), 2))
        self.W_nz_list[i].append(np.count_nonzero(W>(1.0/self.num_region)))
        self.opt_list[i].append(round(opt, 2))

    def update_ucb(self, bandit, arms, epoch):
        num_msgs = bandit.num_msgs
        stat = [epoch, arms, num_msgs.copy()]
        ucbs = {}
        j = 0
        for a in arms:
            ucb_entry = bandit.ucb_table[(j,a)]
            scores = list(np.round(ucb_entry.score_list, 2))
            times = list(np.round(ucb_entry.times))
            # shares = list(np.round(ucb_entry.shares))
            times_index = list(np.round(ucb_entry.times_index))
            T_list = list(np.round(ucb_entry.T_list))
            max_time_list = list(np.round(ucb_entry.max_time_list))
            n = ucb_entry.n
            ucbs[j] = (a, scores, times, times_index, T_list, max_time_list, n)
            j+= 1
        stat.append(ucbs)
        self.bandits[bandit.id].append(stat)

    def print_ucb(self):
        with open(self.log_ucb_filename, 'w') as w:
            for i in range(self.num_node):
                w.write('\t\tnode {}\n'.format(i))
                for stat in self.bandits[i]:
                    epoch, arms, num_msgs, ucbs  = stat
                    num_region = len(ucbs)
                    w.write('>epoch {}. arms {}\n'.format(epoch, arms))
                    for l in range(num_region):
                        arm, scores, times, times_index, T_list, max_time_list, n = ucbs[l]
                        w.write('region={region} {num_msg}, arm={arm}, {n} # {times} # {times_index} # {scores} # {T_list} # {max_time_list} \n'.format(
                            region=l, num_msg=num_msgs[l], arm=arm, 
                            n=n, times=times, times_index=times_index, scores=scores, 
                            T_list=T_list, max_time_list=max_time_list))

    def print_WH(self):
        with open(self.log_solver_filename, 'w') as w:
            for i in range(self.num_node):
                w.write('node' + str( i) + '\n')
                H_ =  self.H_list[i]
                for e in range(len(H_)):
                    H = H_[e]
                    w.write('epoch ' +str(e) + '\n')
                    for i in range(H.shape[0]):
                        t = [str(r) for r in list(np.round(H[i], 2))]
                        txt = ' '.join(t)
                        w.write('[' +txt + ']\n')


                # w.write('\t\tnode {}\n'.format(i))
                # w.write('H mean {}\n'.format(self.H_mean_list[i]))
                # w.write('H max {}\n'.format(self.H_max_list[i]))
                # w.write('H zero {}\n'.format(self.H_nz_list[i]))
                # # w.write('W mean {}\n'.format(self.W_mean_list[i]))
                # # w.write('W max {}\n'.format(self.W_max_list[i]))
                # w.write('W > 1/L {}\n'.format(self.W_nz_list[i]))
                # w.write('opt {}\n'.format(self.opt_list[i]))
         






