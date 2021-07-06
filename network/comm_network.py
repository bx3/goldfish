import sys
from config import MISMATCH
import numpy as np
import random 
from collections import namedtuple
from collections import defaultdict

class NodeState:
    def __init__(self, received, recv_time, from_whom, views, peers, node_delay):
        self.received = received
        self.recv_time = recv_time
        self.from_whom = from_whom
        self.views = views
        self.peers = peers
        self.node_delay = node_delay

        # self.views_hist = defaultdict(list)

def get_broadcast_node(node_hash):
    hash_sum = np.sum(node_hash)
    r = random.random() * hash_sum
    for u, hash_value in enumerate(node_hash):
        if r > hash_value:
            r -= hash_value
        else:
            return u
    return len(node_hash) - 1


def print_debug(i, node, v, peer, ld, time_table):
    #if node.views[v] <  -1 * MISMATCH:
    t, direction = time_table[i][v][-1]
    if t < -1 * MISMATCH:
        print(node.views[v] )
        print('node',i, 'from', node.from_whom, 'recv_time', node.recv_time)
        print('peer', v, 'recv_time', peer.recv_time)
        print('ld[peer][node]', ld[v][i])
        print('ld[node][peer]', ld[i][v])
        print('node', node.node_delay)
        print('peer', peer.node_delay)
        print("peer.recv_time", peer.recv_time)
        print("sum", peer.recv_time + node.node_delay + ld[v][i])
        print("node.recv_time", node.recv_time)
        print()
        sys.exit(1)


def fuzzy_greater(a, b, mismatch):
    if a - b > mismatch:
        return True 
    else:
        return False 

# u is the broadcasting node
def init_comm_network(nodes, u):
    network = {}
    for i in range(len(nodes)):
        peers = nodes[i].get_peers() # both ins and outs
        node_delay = nodes[i].node_delay
        if u != i:
            network[i] = NodeState(False, 0.0, None, {}, peers, node_delay)
        else:
            network[i] = NodeState(True, 0.0, u, {}, peers, node_delay)
    return network

# ld is link delau, bd is broadcasting node, 
# output time_tables is a nested dictionary
# in the first layer, key is node id, I, value is another dict
# in the second lauer, key is the node id who sent msg to I, value is the time
def broadcast_msg(u, nodes, ld, time_tables, abs_time_tables):
    graph = init_comm_network(nodes, u)
    
    broad_nodes = [u]
    while len(broad_nodes) > 0:
        u = broad_nodes.pop(0)
        node = graph[u]
        assert(node.received) # a node must have recved to broadcast
        is_updated = False
        for v in node.peers:
            if v != u:
                peer = graph[v]
                if not peer.received:
                    # if dst has not received it
                    peer.recv_time = node.recv_time + ld[u][v] + peer.node_delay 
                    peer.received = True
                    peer.from_whom = u
                    broad_nodes.append(v)
                else:
                    # if dst has recved, check if it is possible that route is earlier
                    t = peer.recv_time + ld[v][u] + node.node_delay
                    if fuzzy_greater(node.recv_time, t, MISMATCH):
                        node.recv_time = t
                        node.from_whom = v
                        is_updated = True
        # if that route is earlier, check if my all my peers can have earlier time by that route
        if is_updated:
            for v in node.peers:
                peer = graph[v]
                if fuzzy_greater(
                    peer.recv_time, 
                    node.recv_time + ld[u][v] + peer.node_delay,
                    MISMATCH
                ):
                    broad_nodes.insert(0, v)

    # find relative times for each node
    # for i, node in nodes.items():
        # for v in node.get_peers():
    for i, node in graph.items():
        # abs_time_tables[i][i].append(node.recv_time) 
        for v in node.peers:
            # peer = nodes[v]
            peer = graph[v]
            rel_time = peer.recv_time + node.node_delay + ld[v][i] - node.recv_time
            if rel_time < MISMATCH:
                rel_time = 0

            direction=None
            # if v is in both in and out. Consider it as an outgoing
            if v in nodes[i].outs:
                direction = 'outgoing'
            elif v in nodes[i].ins:
                direction = 'incoming'
            else:
                print("unknown direction")
                sys.exit(1)

            if peer.from_whom != i:
                time_tables[i][v].append((rel_time, direction)) #node.views[v]
                abs_time_tables[i][v].append((peer.recv_time + node.node_delay + ld[v][i], direction))
                # safety check
                print_debug(i, node, v, peer, ld, time_tables)
            else:
                time_tables[i][v].append((None, direction)) #node.views[v] rel_time
                abs_time_tables[i][v].append((None, direction))
