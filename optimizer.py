import sys
import numpy as np
import copy
# from sklearn.decomposition import NMF
import time
import config
import solver

class Optimizer:
    def __init__(self, node_id, num_node, num_region, window):
        self.table = [] # raw relative time records, each element is a dict:peer -> time list
        self.abs_table = []
        
        self.id = node_id
        self.N = num_node
        self.L = num_region # region = num out degree
        self.T = window # time window that discard old time data
        self.X = None #np.zeros(self.T, self.N) # array of array, col is num node, row is time
        self.window = window

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

        W, H = solver.run_pgd_nmf(self.id, self.table[-self.window:], self.window, self.N, self.L)
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
        return W, H

    def print_matrix(self, W):
        for i in range(W.shape[0]):
            print(list(W[i]), sum(W[i]))

    def argsort_peers(self, H):
        L_neighbors = np.argsort(H, axis=1)
        return L_neighbors


    # takes best neighbors from matrix B, currently using argmin, later using bandit
    def choose_best_neighbor(self, H):
        L_neighbors = np.argmin(H, axis=1)
        #for i in range(len(L_neighbors)):
        #    print(list(np.round(B[i,:],2)))
        #print('L_neighbors', L_neighbors)
        outs_conn = set(L_neighbors)
        return list(outs_conn)



# sklearn nmp 
# model = NMF(n_components=self.L, init='nndsvd', random_state=0, max_iter=config.max_iter)
# A = model.fit_transform(X)
# B = model.components_

