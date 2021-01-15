import numpy as np
from collections import defaultdict
import random

class Note:
    def __init__(self, nid, n_delay, in_lim, out_lim, init_outs): 
        self.ins = set() 
        self.outs = set(init_outs) 
    
        self.id = nid
        self.node_delay = n_delay
        self.in_lim =  in_lim
        self.out_lim = out_lim 

        self.received = False
        self.from_whom = None # for debug 
        self.recv_time = 0

        # every epoch, a msg is broadcasted and hence values equals to number of message
        # self.views = {} # key is peer id, value is current relative time 
        # self.views_hist = defaultdict(list) # key is peer id, value is a list of time for all sub round
        # self.prev_score = {} #key is combination, value is previous score
        # self.seen = set(init_outs)

        # self.num_in_request = 0

    def get_peers(self):
        return self.outs | self.ins # get union
