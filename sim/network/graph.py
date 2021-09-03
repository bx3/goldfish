import sys
from network.communicator import Communicator
from network.oracle import SimpleOracle

class Graph:
    def __init__(self, num_node, proc_delay, num_in, num_out):
        self.num_node = num_node
        self.proc_delay = proc_delay
        self.nodes = {i: Communicator(i, self.proc_delay[i], num_in, num_out, []) 
                for i in range(self.num_node)}
        self.oracle = SimpleOracle(num_in, num_out, self.num_node)


