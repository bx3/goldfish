import sys
import numpy as np
import copy
import time
import config

class Optimizer:
    def __init__(self, node_id, num_cand, num_region, window, batch_type):
        self.table = [] # raw relative time records, each element is a dict:peer -> time list
        
        self.id = node_id
        self.N = num_cand
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
