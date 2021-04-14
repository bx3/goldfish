import sys
import numpy as np
import copy
# from sklearn.decomposition import NMF
import time
import config
import solver

class SparseTable:
    def __init__(self, node_id, num_node, num_region, window):
        self.table = []
        self.id = node_id
        self.N = num_node
        self.L = num_region # region != num out degree

    def append_time(self, slots, num_msg):
        lines = [[] for _ in range(num_msg)] 
        i = 0
        for p, t_list in slots.items():
            if len(t_list) != num_msg:
                print('Error. append_time sparse table')
                print(len(t_list))
                print(num_msg)
                sys.exit(2)
            # debug
            # if None in t_list:
                # print(t_list)
            for i in range(num_msg):
                if t_list[i] != None:
                    lines[i].append((p, t_list[i])) 
        for i in range(num_msg):
            self.table.append(lines[i])

class Optimizer:
    def __init__(self, node_id, num_node, num_region, window, batch_type):
        self.table = [] # raw relative time records, each element is a dict:peer -> time list
        
        self.id = node_id
        self.N = num_node
        self.L = num_region # region = num out degree
        self.T = window # time window that discard old time data
        self.X = None #np.zeros(self.T, self.N) # array of array, col is num node, row is time

        if batch_type == 'append':
            self.window = 0 
        else:
            self.window = window

        self.batch_type = batch_type

        self.H_est = None
        self.W_est = None
        self.H_prev = None
        self.W_prev = None
        self.H_input = None
        self.W_input = None
        self.H_mean = None
        self.H_mean_mask = None

        # testing easy case, dc 
        self.H_truth = np.zeros((self.L, self.N))
        num_node_per_region = int((self.N-1)/self.L)
        for i in range(self.L):
            for j in range(1, self.N):
                if int((j-1) / num_node_per_region) == i:
                    self.H_truth[i,j] = 20
                else:
                    self.H_truth[i,j] = 200
        # self.H_input = self.H_truth.copy()



    # slot can be either rel time table or abs time table
    def append_time(self, slots, num_msg):
        lines = [[] for _ in range(num_msg)] 
        i = 0
        for p, t_list in slots.items():
            assert(len(t_list) == num_msg)
            for i in range(num_msg):
                lines[i].append((p, t_list[i])) 
        for i in range(num_msg):
            self.table.append(lines[i])
    
    def store_WH(self, W, H):
        self.W_est = W.copy()
        self.H_est = H.copy()
        self.W_input = W.copy()
        if self.W_prev is None:
            self.W_prev = W.copy()
        if self.H_prev is None:
            self.H_prev = H.copy()

    def get_new_W(self, W_est, num_msg):
        if self.batch_type == 'rolling':
            print("self.T", self.T, "num_msg", num_msg)
            W_out = np.zeros(W_est.shape)
            new_noise = np.random.rand(num_msg, W_est.shape[1])
            W_out[:self.T-num_msg] = W_est[num_msg:]
            W_out[self.T-num_msg:] = new_noise
        elif self.batch_type == 'append':
            new_noise = np.random.rand(num_msg, W_est.shape[1])
            W_out = np.vstack((W_est, new_noise))
        else:
            print("Unknown batch_type", self.batch_type)
            sys.exit(1)
        return W_out

    def get_new_H(self, H_mean, H_mean_mask):
        mean =  np.sum(H_mean*H_mean_mask)/np.sum(H_mean_mask)
        H_new = H_mean*H_mean_mask + mean*(H_mean_mask!=1)
        return H_new 

    def update_WH(self, num_msg):
        self.W_input = self.get_new_W(self.W_prev, num_msg)
        self.H_input = self.get_new_H(self.H_mean, self.H_mean_mask)

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

