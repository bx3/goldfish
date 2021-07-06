#!/usr/bin/env python
import matplotlib.pyplot as plt
import json
import sys

if len(sys.argv) < 2:
    print('need json topology file, show pub for continent cluster')
    print('need output path')
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]

nodes = []
summary = None
with open(infile) as f:
    data = json.load(f)
    nodes = data['nodes']
    summary = data['summary']
print(summary)
square_len = int(summary['square_length'])
x_list = []
y_list = []
names = []
pub_x = []
pub_y = []
proc_delay_list = []
pub_delay_list = []
for u in nodes:
    u_id = int(u["id"])
    x = float(u["x"])
    y = float(u["y"])
    proc_delay = float(u["proc_delay"])

    role = u["role"]
    if role == "PUB":
        pub_x.append(x)
        pub_y.append(y)
        pub_delay_list.append(proc_delay)

    proc_delay_list.append(proc_delay)
    x_list.append(x)
    y_list.append(y)
    names.append(u_id)

default_size = 3
lim_size = 30
num_size = 10

pub_delay_list = [int(i-min(proc_delay_list))+default_size for i in pub_delay_list]
proc_delay_list = [int(i-min(proc_delay_list))+default_size for i in proc_delay_list]
print(proc_delay_list)
print(pub_delay_list)

fig, ax = plt.subplots()
ax.scatter(x_list, y_list,s=proc_delay_list)
pub_size = []
ax.scatter(pub_x, pub_y, s=pub_delay_list, color='red')
ax.set_xlim(0, square_len)
ax.set_ylim(0, square_len)

for i in range(len(nodes)):
    ax.annotate(names[i], (x_list[i], y_list[i]))

plt.savefig(outfile)
