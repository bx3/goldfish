#!/usr/bin/env python
import sys
import os
from collections import defaultdict
if len(sys.argv) < 2:
    print('filename')
    sys.exit(1)

filename = sys.argv[1]
conn_ins = defaultdict(list)
conn_outs = {}
num_node = 0
with open(filename) as f:
    for line in f:
        num_node += 1
        tokens = line.strip().split()
        node = int(tokens[0])
        conn_outs[node] = [int(i) for i in tokens[1:]]

for i in range(num_node):
    outs = conn_outs[i]
    for p in outs:
        conn_ins[p].append(i)

#for i in range(num_node):


print(0)
print('num out', conn_outs[0])
print('num in ', conn_ins[0])
for p in conn_ins[0]:
    print(len(conn_ins[p]))

