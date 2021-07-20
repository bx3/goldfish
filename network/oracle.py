from collections import namedtuple
from collections import defaultdict 
import config
import sys
import copy

# messages that a node has information about other nodes, all two, three hops contains RecNote
PeersInfo = namedtuple('PeersInfo', ['one_hops', 'two_hops', 'three_hops']) 
RecNote = namedtuple('RecNote', ['rec', 'id']) # rec for recommender id, id is the node

class NodeConn:
    def __init__(self, i, ins, outs):
        self.id = i
        self.ins = set(ins.copy())
        self.outs = set(outs.copy())
    def update_ins(self, add_ins, rm_ins):
        assert(len(self.ins.difference(add_ins)) == 0 ) # no repeat adds
        self.ins = self.ins.union(add_ins)
        assert(len(self.ins.union(rm_ins)) == len(self.ins) ) # must contain
        self.ins = self.ins.difference(rm_ins)
    def update_outs(self, add_outs, rm_outs):
        assert(len(self.outs.difference(add_outs)) == 0 ) # no repeat adds
        self.outs = self.outs.union(add_outs)
        assert(len(self.outs.union(rm_outs)) == len(self.outs) ) # must contain
        self.outs = self.outs.difference(rm_outs)

class SimpleOracle:
    def __init__(self, in_lim, out_lim, num_node):
        self.in_lim = in_lim
        self.out_lim = out_lim
        self.nodes = {i: NodeConn(i, [], []) for i in range(num_node)} 
        self.claim = {i: None for i in range(num_node)}
    
    def setup(self, curr_conns):
        for i, outs in curr_conns.items():
            self.nodes[i].update_outs(outs, [])
            for j in outs:
                self.nodes[j].update_ins([i], [])

    # check if others are happy with i's connection
    # does not check i's condition
    def can_i_connect(self, i, add_outs):
        unconnectable = []
        for j in add_outs:
            node_j = copy.deepcopy(self.nodes[j])
            node_j.update_ins([i], [])
            if len(node_j.ins) > self.in_lim:
                unconnectable.append(j)
        if len(unconnectable) == 0:
            self.claim[i] = tuple(add_outs)

        return unconnectable

    # node has right to rm both incoming and outgoing
    # node i might be able to add outoing connections provided j's approval
    # node i has not right to compel others nodes to connect to itself (no add_ins)
    def update(self, i, rm_ins, add_outs, rm_outs):
        # only update those already claimed and approved
        if self.claim[i] is not None:
            claimed_add_outs = list(self.claim[i])
            self.claim[i] = None
            assert(claimed_add_outs == add_outs)

            self.nodes[i].update_ins([], rm_ins)
            self.nodes[i].update_outs(add_outs, rm_outs)
            if len(node.ins) > self.in_lim:
                print(i, 'cannot update in', )
                sys.exit(1)
            if len(node.outs) > self.out_lim:
                print(i, 'cannot make change')
                sys.exit(1)

            self.make_change(node_i, [], rm_ins, add_outs, rm_outs)
            for j in add_outs:
                self.nodes[j].update_ins([i], [])
            for j in rm_outs:
                self.nodes[j].update_ins([], [i])
        else:
            print(i, 'update without claim')
            sys.exit(1)

    # return T/F to approve the update
    # def claim(self, i, rm_ins, add_outs, rm_outs):
        # node_i = copy.deepcopy(self.nodes[i])
        # # node i has not right to compel others nodes to conn to itself
        # if not self.make_change(node_i, [], rm_ins, add_outs, rm_outs):
            # return False
        # # added j by node i should comply to the change
        # for j in add_outs:
            # node_j = copy.deepcopy(self.nodes[j])
            # if not self.make_change(node_j, [i], [], [], []):
                # return False
        # for j in rm_ins:
            # node_j = copy.deepcopy(self.nodes[j])
            # if not self.make_change(node_j, [], [], [], [i]):
                # return False


        # self.claim[i] = (add_ins, rm_ins, add_outs, rm_outs)
        # return True

    

class NetworkOracle:
    def __init__(self, is_dynamic, sybils, selectors):
        self.outs_keeps = {} # key is node id, value is a list of keep ids
        self.conn_2 = {} # key is node, it is confirmed direct connection
        self.conn_3 = {}
        self.is_dynamic = is_dynamic
        self.sybils = sybils
        self.selectors = selectors


    def update_1_hop_peers(self, u, peers):
        assert(u not in self.outs_keeps)
        self.outs_keeps[u] = list(peers)

    # used when peers set 2hop peers
    def update_2_hop_peers(self, u, peers):
        if u in self.conn_2:
            print(u, 'already in conn_2', self.conn_2)
            sys.exit(1)
        self.conn_2[u] = peers.copy()

    def update_3_hop_peers(self, u, peers):
        if u in self.conn_3:
            print(u, 'already in conn_3', self.conn_3)
            sys.exit(1)
        self.conn_3[u] = peers.copy()

    # get one hop peers for that node, which include keep and newly added 2info node
    def get_1_hop_peers(self, v):
        if not self.is_dynamic:
            if config.recommend_worst_attack:
                if v in self.sybils:
                    # give worst neighbors, TODO
                    assert(self.selectors[v].worst_compose != None)
                    return self.selectors[v].worst_compose.copy()
                else:
                    return self.outs_keeps[v].copy() # copy
            else:
                # honest case
                # print('honest', self.outs_keeps[v].copy())
                return self.outs_keeps[v].copy() # copy
        else:
            # print('is dynamic')
            return self.outs_keeps[v].copy() + self.conn_2[v].copy()

    # used to give nodes info
    def get_multi_hop_info(self, u):
        # all honest case
        one_hops = self.outs_keeps[u].copy()
        two_hops_map = {} # key is 1hop, values are 2hops
        three_hops_map = {} # key is 2hop, values 3hops
        for v in one_hops:
            peers = self.get_1_hop_peers(v)
            two_hops_map[v] = peers
        for v, two_hops in two_hops_map.items():
            
            for w in two_hops:
                if w not in three_hops_map:
                    peers = self.get_1_hop_peers(w)
                    three_hops_map[w] = peers
        return PeersInfo(one_hops, two_hops_map, three_hops_map)
