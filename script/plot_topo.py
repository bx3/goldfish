#!/usr/bin/env python
import matplotlib.pyplot as plt
import json
import sys

def plot_topology(infile, ax, interested):
    nodes = []
    summary = None
    with open(infile) as f:
        data = json.load(f)
        nodes = data['nodes']
        summary = data['summary']
        topo_type = summary['topo_type']
    # square_len = int(summary['square_length'])
    x_list = []
    y_list = []
    names = []
    pub_x = []
    pub_y = []
    int_x = []
    int_y = []

    proc_delay_list = []
    pub_delay_list = []
    int_delay_list = []
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
        if u_id in interested:
            int_x.append(x)
            int_y.append(y)
            int_delay_list.append(proc_delay)

        proc_delay_list.append(proc_delay)
        x_list.append(x)
        y_list.append(y)
        names.append(u_id)

    # ploting dot size
    default_size = 3
    lim_size = 30
    num_size = 10

    pub_delay_list = [int(i-min(proc_delay_list))+default_size+5 for i in pub_delay_list]
    proc_delay_list = [int(i-min(proc_delay_list))+default_size for i in proc_delay_list]
    int_delay_list = [int(i-min(proc_delay_list))+default_size for i in int_delay_list]


    ax.scatter(x_list, y_list,s=proc_delay_list)
    pub_size = []
    ax.scatter(pub_x, pub_y, s=pub_delay_list, color='red')
    ax.scatter(int_x, int_y, s=int_delay_list, color='orange')

    if topo_type == 'rand':
        square_len = int(summary['square_length'])
        ax.set_xlim(0, square_len)
        ax.set_ylim(0, square_len)
    elif topo_type == 'real':
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    else:
        print('Unknown topo type', topo_type)


    for i in range(len(nodes)):
        ax.annotate(names[i], (x_list[i], y_list[i]))
    return x_list, y_list 


def plot_graph(topo_json, graph_json, ax, stars, pubs):
    x_list, y_list = plot_topology(topo_json, ax, stars)
    graph = None
    with open(graph_json) as f:
        data = json.load(f)
        graph = data
        
    for node in graph:
        i = node['node']
        line_x = [x_list[i]]
        line_y = [y_list[i]]
        for o in node['outs']:
            if i in stars:
                ax.plot(line_x+[x_list[o]], line_y+[y_list[o]], 'b-')
            elif i in pubs:
                ax.plot(line_x+[x_list[o]], line_y+[y_list[o]], 'r-')
            # else:
                # ax.plot(line_x+[x_list[o]], line_y+[y_list[o]], 'k-')




def __init__():
    if len(sys.argv) < 3:
        print('need json topology file, show pub for continent cluster')
        print('need output path')
    sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    interested = []

    if len(sys.argv) >=3:
        interested = [int(i) for i in sys.argv[3:]]
    fig, ax = plt.subplots()
    plot_topology(infile, ax, interested)

    plt.savefig(outfile)
