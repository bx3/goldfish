import sys
import numpy as np
import copy
from sklearn.decomposition import NMF
import time
import config
import solver

class Optimizer:
    def __init__(self, node_id, num_node, num_region, window, pools):
        self.table = [] # raw relative time records, each element is a dict:peer -> time list
        self.abs_table = []
        
        self.id = node_id
        self.N = num_node
        self.L = num_region # region = num out degree
        self.T = window # time window that discard old time data
        self.X = None #np.zeros(self.T, self.N) # array of array, col is num node, row is time
        self.window = window
        self.pools = None # pools

    # slot can be either rel time table or abs time table
    def append_time(self, slot):
        self.table.append(list(slot.items()))

    def construct_table(self):
        X = np.zeros((self.window, self.N)) 
        i = 0 # row

        max_time = 0 
        for slot in self.table[-self.window:]:
            for p, t in slot:
                if t[0] < 0:
                    print('time', t)
                    sys.exit(1)
                X[i, p] = t[0] # t is a single element list, if num_msg is 1
                if t[0] > max_time:
                    max_time = t[0]
            i += 1
        return X

    # return matrix B, i.e. region-node matrix that containing real value score
    def matrix_factor(self):
        # sample time from each table, to assemble the matrix X
        X = self.construct_table()

        # print(X.shape)
        # for i in range(X.shape[0]):
            # entries = []
            # for t in X[i]:
                # if t > 0:
                    # entries.append(t)
            # print(self.id, sum(X[i]> 0), min(entries), X[i][self.id])
            # print(X[i])
        # sys.exit(2)

        W, H = solver.run_pgd_nmf(self.L, X)
        # self.print_matrix(W)
        
        # sys.exit(2)

        # then use best neighbor methods to select get neighbors
        out_conns = self.choose_best_neighbor(H)
        return out_conns

    def print_matrix(self, W):
        for i in range(W.shape[0]):
            print(list(W[i]))


    # takes best neighbors from matrix B, currently using argmin, later using bandit
    def choose_best_neighbor(self, B):
        L_neighbors = np.argmin(B, axis=1)
        #for i in range(len(L_neighbors)):
        #    print(list(np.round(B[i,:],2)))
        #print('L_neighbors', L_neighbors)
        outs_conn = set(L_neighbors)
        return list(outs_conn)



# sklearn nmp 
# model = NMF(n_components=self.L, init='nndsvd', random_state=0, max_iter=config.max_iter)
# A = model.fit_transform(X)
# B = model.components_

