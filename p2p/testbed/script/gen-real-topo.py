#!/usr/bin/env python
import sys
from collections import namedtuple
import pandas as pd
import numpy as np
import time
import json
ServerMeta = namedtuple('ServerMeta', ['id', 'name', 'title', 'location', 'state','country','state_abbv','continent','lat','long'])
PingData = namedtuple('PingData', ['src', 'dst', 'ts', 'min', 'avg', 'max', 'dev'])

if len(sys.argv) < 9:
    print("Require args. ")
    sys.exit(1)

server_meta_csv = sys.argv[1]
ping_csv = sys.argv[2]
num_node = int(sys.argv[3])
num_pub = int(sys.argv[4])
name = sys.argv[5]
proc_delay_mean = float(sys.argv[6])
proc_delay_std = float(sys.argv[7])
seed = int(sys.argv[8])
np.random.seed(seed)

def parse_server_meta(filename):
    data = pd.read_csv(filename)
    servers = {}
    for i, sid_ in data['id'].items():
        sid = data['id'][i]
        assert(sid_== sid)
        name = data['name'][i]
        title = data['title'][i]
        location = data['location'][i]
        state = data['state'][i]
        country = data['country'][i]
        state_abbv = data['state_abbv'][i]
        continent = data['continent'][i]
        lat = data['latitude'][i]
        lon = data['longitude'][i]
        servers[sid] = ServerMeta(sid, name, title, location, state, country, state_abbv, continent, lat, lon)
    return servers

def parse_ping_data(filename, nodes_ind):
    ping_data = {}
    with open(filename) as f:
        next(f)
        for line in f:
            tokens = line.strip().strip('"').split('","')
            src = int(tokens[0])
            dst = int(tokens[1])
            if nodes_ind[src]==1 and nodes_ind[dst]==1:
                ts = tokens[2]
                min_t = float(tokens[3])
                avg = float(tokens[4])
                max_t = float(tokens[5])
                dev = float(tokens[6])
                if (src, dst) not in ping_data:
                    ping_data[src, dst] = PingData(src, dst, ts, min_t, avg, max_t, dev)
    return ping_data
    # data = pd.read_csv(filename)
    # ping_data = {}
    # for i, src_ in data['source'].items():
        # src = data['source'][i]
        # assert(src == src_)
        # dst = data['destination'][i]
        # if src in nodes or dst in nodes:
            # ts = data['timestamp'][i]
            # min_t = data['min'][i]
            # avg = data['avg'][i]
            # max_t = data['max'][i]
            # dev = data['mdev'][i]
            # ping_data[src] = PingData(src, dst, ts, min_t, avg, max_t, dev)
    # return ping_data

server_meta = parse_server_meta(server_meta_csv)
num_server_meta = len(server_meta)
if num_server_meta < num_node:
    print("real data insufficent")
    sys.exit(1)
all_nodes = list(server_meta.keys())

nodes_ind = np.zeros(1000)
nodes_ind[all_nodes] = 1
ping_data = parse_ping_data(ping_csv, nodes_ind)

node_missing = {}
for i in all_nodes:
    num_data = 0
    missing = []
    for j in all_nodes:
        if i != j:
            if (i,j) in ping_data or (j,i) in ping_data:
                num_data+=1
            else:
                missing.append(j)
    node_missing[i] = missing
sorted_node_missing =  sorted(node_missing.items(), key=lambda item: len(item[1]), reverse=True)

removed = set()
for i, missing in sorted_node_missing:
    for j in missing:
        if j in all_nodes:
            all_nodes.remove(i)
            removed.add(i)
            break

nodes = sorted(np.random.choice(all_nodes, num_node, replace=False))
pubs = np.random.choice(nodes, num_pub, replace=False)

topo_nodes = []

dataId_to_topoId = {}
j = 0
for i in nodes:
    dataId_to_topoId[i] = j
    j += 1

for i in nodes:
    adj = []
    for j in nodes:
        if i != j:
            if (i,j) in ping_data:
                adj.append(ping_data[i,j].avg)
            else:
                adj.append(ping_data[j,i].avg)

        else:
            adj.append(0)

    role = 'LURK'
    if i in pubs:
        role = 'PUB'
    proc_delay = np.random.normal(proc_delay_mean, proc_delay_std)
    node = {
        'id': dataId_to_topoId[i],
        'center': None,
        'role': role,
        'x': int(server_meta[i].long),
        'y': int(server_meta[i].lat),
        'proc_delay': proc_delay,
        'adj': adj 
        }
    topo_nodes.append(node)

summary = {
    'num_node': int(num_node),
    'num_pub': int(num_pub),
    'topo_type': 'real'
    }
setup = {'summary': summary, 'nodes': topo_nodes}

print(json.dumps(setup, indent=4))
