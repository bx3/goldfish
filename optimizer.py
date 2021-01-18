import sys
import numpy as np
import copy
# from sklearn.decomposition import NMF
import time
import config
import solver
import nndsvd
from scipy.sparse.linalg import svds

class SparseTable:
    def __init__(self, node_id, num_node, num_region, window):
        self.table = []
        self.id = node_id
        self.N = num_node
        self.L = num_region # region = num out degree
        self.window = window # time window that discard old time data

    def append_time(self, slots, num_msg):
        lines = [[] for _ in range(num_msg)] 
        i = 0
        for p, t_list in slots.items():
            assert(len(t_list) == num_msg)
            for i in range(num_msg):
                lines[i].append((p, t_list[i])) 
        for i in range(num_msg):
            self.table.append(lines[i])

def worker(worker_id, sink, task_source, node_list, num_node, num_region, window):
    optimizers = {}
    for i in node_list:
        optimizers[i] = Optimizer(i, num_node, num_region, window)
    # keep working until done
    while True:
        task = task_source.recv()
        # all jobs finish
        if task == None:
            sink.send(None)
            sink.close()
            task_source.close()
            break
        node, sparse_table, new_msg_start = task
        optimizers[node].matrix_factor(sparse_table, new_msg_start)
        print(node, 'before send')
        sink.send((node, optimizers[node].W, optimizers[node].H)) # node, W, H  
        print(node, 'after send')
        time.sleep(0.15) # time for msg broadcasting in next step


class Optimizer:
    def __init__(self, node_id, num_node, num_region, window):
        self.id = node_id
        self.N = num_node
        self.L = num_region # region = num out degree
        self.T = window # time window that discard old time data

        self.window = window
        self.H = None
        self.W = None
        self.is_init = True

    def matrix_factor(self, sparse_table, new_msg_start):
        X, max_time = solver.construct_table(self.N, sparse_table)
        A, B = None, None
        if not config.feedback_WH or self.is_init: 
            self.is_init = False
            print(self.id, 'init')
            if config.init_nndsvd:
                A, B = nndsvd.initial_nndsvd(X, self.L, config.nndsvd_seed)
            else:
                A, S, B = svds(X, L)
                I = np.sign(A.sum(axis=0)) # 2 * int(A.sum(axis=0) > 0) - 1
                A = A.dot(np.diag(I))
                B = np.transpose((B.T).dot(np.diag(S*I)))
        else:
            print(self.id, 'feedback')
            shape = (self.T, self.L)
            A = np.zeros(shape)
            A[:new_msg_start] = self.W[:new_msg_start] + np.random.normal(
                        config.W_noise_mean, 
                        config.W_noise_std, size=(new_msg_start, self.L)) 
            A[new_msg_start:] = np.ones((self.T-new_msg_start, self.L))

            H_mean = np.mean(self.H)
            H_std = np.std(self.H)
            B = self.H + np.random.normal(H_mean, H_std/config.H_noise_std_ratio, size=(self.L, self.N))

        self.W, self.H = solver.alternate_minimize(A, B, X, self.L)

    # return matrix B, i.e. region-node matrix that containing real value score
    # def matrix_factor(self):
        # sample time from each table, to assemble the matrix X
        # X = self.construct_table()
        # if self.id == 32:
            # print(self.id, X[np.nonzero(X)])

        # print(X.shape)
        # for i in range(X.shape[0]):
            # entries = []
            # for t in X[i]:
                # if t > 0:
                    # entries.append(t)
            # print(self.id, sum(X[i]> 0), min(entries), X[i][self.id])
            # print(X[i])
        # sys.exit(2)

        # W, H = solver.run_pgd_nmf(self.id, self.table[-self.window:], self.window, self.N, self.L)
        # self.print_matrix(W)
        # print('')
        # then use best neighbor methods to select get neighbors
        # out_conns = self.choose_best_neighbor(H)

        # tie is not randomized
        # sorted_conns = self.choose_best_neighbor(H) #argsort_peers
        # for i in range(sorted_conns.shape[0]):
            # if sorted_conns[i][0] != np.argmin(H[i]):
                # print('num zero', sum(H[i] == 0))
                # print(list(np.round(H[i],2)))
                # print(sorted_conns[i])
                # print(np.argmin(H[i]))
                # print(min(H[i]))
                # print(H[i][np.argmin(H[i])])
                # print(H[i][sorted_conns[i][0]])
                # sys.exit(2)

        # print(sorted_conns)
        # return W, H

    # def print_matrix(self, W):
        # for i in range(W.shape[0]):
            # print(list(W[i]), sum(W[i]))

    # def argsort_peers(self, H):
        # L_neighbors = np.argsort(H, axis=1)
        # return L_neighbors


    # takes best neighbors from matrix B, currently using argmin, later using bandit
    # def choose_best_neighbor(self, H):
        # L_neighbors = np.argmin(H, axis=1)
        # #for i in range(len(L_neighbors)):
        # #    print(list(np.round(B[i,:],2)))
        # #print('L_neighbors', L_neighbors)
        # outs_conn = set(L_neighbors)
        # return list(outs_conn)



# sklearn nmp 
# model = NMF(n_components=self.L, init='nndsvd', random_state=0, max_iter=config.max_iter)
# A = model.fit_transform(X)
# B = model.components_

