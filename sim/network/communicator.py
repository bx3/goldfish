class Communicator:
    def __init__(self, nid, n_delay, in_lim, out_lim, out_conns): 
        self.ins =set()
        # self.ordered_outs = out_conns 
        self.outs = set(out_conns) 
    
        self.id = nid
        self.node_delay = n_delay
        self.in_lim =  in_lim
        self.out_lim = out_lim 

        # self.received = False
        # self.from_whom = None # for debug 
        # self.recv_time = 0

    def get_peers(self):
        return self.outs | self.ins # get union

    def update_conns(self, outs, ins):
        self.outs = set(outs)
        self.ins = set(ins)
        
