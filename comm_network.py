import sys
from config import MISMATCH
import numpy as np
import random 
from collections import namedtuple
from collections import defaultdict

# NodeState = namedtuple('NodeState', ['received', 'recv_time', 'from_whom', 'views', 'peers', 'node_delay']) 
class NodeState:
    def __init__(self, received, recv_time, from_whom, views, peers, node_delay):
        self.received = received
        self.recv_time = recv_time
        self.from_whom = from_whom
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
    if time_table[i][v][-1] < -1 * MISMATCH:
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

# ld is link delau, bd is broadcasting node, nh is node_hash
def broadcast_msg(u, nodes, ld, nh, time_tables, abs_time_tables):
    graph = init_comm_network(nodes, u)
    # precondition
    # for i, node in nodes.items():
        # node.received = False
        # node.recv_time = 0
        # node.views = {}
        # node.from_whom = None

    # nodes[u].received = True
    # nodes[u].recv_time = 0
    # nodes[u].from_whom = u

    broad_nodes = [u]
    while len(broad_nodes) > 0:
        u = broad_nodes.pop(0)
        node = graph[u]
        # node = nodes[u]
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

            if peer.from_whom != i:
                # node.views[v] = peer.recv_time + node.node_delay + ld[v][i] - node.recv_time
                time_tables[i][v].append(rel_time) #node.views[v]
                abs_time_tables[i][v].append(peer.recv_time + node.node_delay + ld[v][i])
                # safety check
                print_debug(i, node, v, peer, ld, time_tables)
            else:
                # TODO should return None
                time_tables[i][v].append(rel_time) #node.views[v] rel_time
                abs_time_tables[i][v].append(None)


            # make sure the node actually transmit to me
            # if node.views[v] >= ld[i][v] + ld[v][i] + node.node_delay + peer.node_delay:
                # node.views[v] = unlimit

