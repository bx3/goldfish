#!/usr/bin/env python
import sys
import json
import numpy as np

if len(sys.argv) < 5:
    print("Rquire. num-node<int> num-pub<int> num-adapt<int> port<int>") # proc-delay-ms<int>
    sys.exit(1)

num_node = int(sys.argv[1])
num_pub = int(sys.argv[2])
num_adapt = int(sys.argv[3])
port = int(sys.argv[4])
# proc_delay = int(sys.argv[4])

all_nodes = [i for i in range(num_node)]
# pubs = np.random.choice(all_nodes, num_pub, replace=False)
pubs = [i for i in range(num_pub)]
adapts = [i for i in range(num_pub, num_pub + num_adapt)]

nodes = {}
for i in range(num_node):
    # pdelay = np.random.randint(0,proc_delay)
    setup = {}
    setup['id'] = i
    setup['addr'] = '127.0.0.1:'+str(port+i)
    if i in pubs:
        setup['prob'] = 1.0 / float(num_pub)
    else:
        setup['prob'] = 0

    if i in adapts:
        setup['adapt'] = True
    else:
        setup['adapt'] = False 

    # setup['proc'] = pdelay
    nodes[i] = setup

for i in range(num_node):
    config = {}
    config['local'] = nodes[i]
    config['peers'] = []
    for j in range(num_node):
        if i != j:
            config['peers'].append(nodes[j])
    filepath = 'configs/node'+str(i)+'.json'
    with open(filepath, 'w') as w:
        json.dump(config, w)
