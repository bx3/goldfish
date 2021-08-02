import sys
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import math

def plot_lats_cdf(lats_pair, ax, title):
    lats = [lat for i, lat in lats_pair]
    cdf = {0:0}
    for i, lat in enumerate(lats):
        cdf[lat] = float(i+1) / len(lats)
    x, y = zip(*cdf.items())
    # print(cdf.items())
    ax.plot(x, y)
    ax.set_title(title)


def plot_diff_lats(all_lats, out_dirs, ax, epochs, xlim, title):
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(epochs))]
    patches = []

    for i in range(len(epochs)):
        p =  mpatches.Patch(color=colors[i], label=str(epochs[i]))
        patches.append(p) 

    tick_spacing = 50
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_prop_cycle('color', colors)

    for e in epochs:
        diff_lat = [x1-x2 for (x1,x2) in zip(all_lats[out_dirs[0]][e], all_lats[out_dirs[1]][e])]
        ax.plot(diff_lat)
    ax.grid(True)
    ax.set_xlim(0, xlim)
    ax.set_title(title, fontsize='small' )
    return patches

# assume epochs are sorted
def plot_figure(percent_X_lats, ax, epochs, min_y, max_y, xlim, title, stars):
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(epochs))]
    patches = []

    for i in range(len(epochs)):
        p =  mpatches.Patch(color=colors[i], label=str(epochs[i]))
        patches.append(p) 

    tick_spacing = 50
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_prop_cycle('color', colors)
    all_sorted_lats = {}
    for e in epochs:
        lats = percent_X_lats[e]
        sorted_node_lats = sorted(lats, key=lambda item: item[1])
        sorted_lats = [lat for i, lat in sorted_node_lats]
        sorted_node = [i for i, lat in sorted_node_lats]
        star_x = []
        star_y = []
        labels = []
        for k in range(len(sorted_node_lats)):
            i, lat = sorted_node_lats[k]
            if i in stars:
                star_x.append(k)
                star_y.append(lat)
                labels.append(i)

        ax.plot(sorted_lats)
        all_sorted_lats[e] = sorted_lats
        ax.scatter(star_x, star_y, marker='x')
        # for i, txt in enumerate(labels):
            # ax.annotate(txt, (star_x[i], star_y[i]))

    ax.grid(True)
    if min_y is not None and max_y is not None:
        ax.set_ylim(min_y, max_y)
    ax.set_xlim(0, xlim)
    ax.set_title(title, fontsize='small')
    return patches, all_sorted_lats

def plot_stars_figure(percent_X_lats, ax, epochs, min_y, max_y, xlim, title, stars):
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(epochs))]
    patches = []

    for i in range(len(epochs)):
        p =  mpatches.Patch(color=colors[i], label=str(epochs[i]))
        patches.append(p) 

    tick_spacing = 50
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_prop_cycle('color', colors)
    epochs_sorted_lats = {}
    for e in epochs:
        lats = percent_X_lats[e]
        stars_lats = [lats[s][1] for s in stars]
        sorted_lats = sorted(stars_lats)
        ax.plot(sorted_lats)
        epochs_sorted_lats[e] = sorted_lats

    ax.grid(True)
    ax.set_ylim(min_y, max_y)
    ax.set_xlim(0, xlim)
    ax.set_title(title, fontsize='small')
    return patches, epochs_sorted_lats


# x is float
def get_Xcent_node(lats, x):
    sorted_lats_pair = sorted(lats.items(), key=lambda item: item[1])
    sorted_lat = [lat for i, lat in sorted_lats_pair]
    if len(sorted_lat) >= 10:
        lat_x = sorted_lat[int(round(len(sorted_lat)*float(x)/100.0)) - 1]
    else:
        lat_x = sorted_lat[int(len(sorted_lat)*float(x)/100.0)]
    return lat_x

def get_Xcent_pubs(lats, x, pubs):
    if x == 'avg':
        lat_list = list(lats.values())
        return sum(lat_list) / float(len(lat_list))
    else:
        sorted_lats_pair = sorted(lats.items(), key=lambda item: item[1])
        sorted_pub_lat = [lat for i, lat in sorted_lats_pair if i in pubs]
        num_pubs = float(len(sorted_pub_lat))
        assert(float(x) >= 0 and float(x) <= 100)
        if len(sorted_pub_lat) >= 10:
            lat_x = sorted_pub_lat[int(round(num_pubs*float(x)/100.0)) - 1]
        else:
            lat_x = sorted_pub_lat[int(num_pubs*float(x)/100.0)]
        return lat_x

def get_diff_Xcent_pubs(i, lats, x, pubs, topo):
    loc, proc_delay = get_topo_loc_delay(topo)

    diff_lats = {}
    for m, lat in lats.items():
        if i != m:
            line_len = (math.sqrt(
                    (loc[i][0]-loc[m][0])**2+
                    (loc[i][1]-loc[m][1])**2 ) + proc_delay[m])
            diff_lats[m] = lat - line_len
        else:
            diff_lats[i] = 0

    if x == 'avg':
        lat_list = list(lats.values())
        return sum(lat_list) / float(len(lat_list))
    else:
        sorted_lats_pair = sorted(diff_lats.items(), key=lambda item: item[1])
        sorted_pub_lat = [lat for i, lat in sorted_lats_pair if i in pubs]
        num_pubs = float(len(sorted_pub_lat))
        assert(float(x) >= 0 and float(x) <= 100)
        if len(sorted_pub_lat) >= 10:
            lat_x = sorted_pub_lat[int(round(num_pubs*float(x)/100.0)) - 1]
        else:
            lat_x = sorted_pub_lat[int(num_pubs*float(x)/100.0)]
        return lat_x



def get_topo_loc_delay(topo_json):
    num_pub = 0
    num_node = None
    loc = {}
    proc_delay = {}
    with open(topo_json) as config:
        data = json.load(config)
        nodes = data['nodes']
        summary = data['summary']
        num_node = summary['num_node']
        for node in nodes:
            loc[node['id']] = (node['x'], node['y'])
            proc_delay[node['id']] = float(node['proc_delay'])

    return loc, proc_delay 



def parse_topo(topo_json):
    num_pub = 0
    num_node = None
    pubs = []
    with open(topo_json) as config:
        data = json.load(config)
        nodes = data['nodes']
        summary = data['summary']
        num_node = summary['num_node']
        role = {}

        for node in nodes:
            if node["role"] == 'PUB':
                num_pub += 1
                pubs.append(node['id'])
    return num_pub, pubs

def parse_adapt(filename):
    adapts = []
    with open(filename) as f:
        for line in f:
            adapt = int(line.split()[0])
            adapts.append(adapt)
    return adapts

# return a list whose i-th entry represnts latency to reach 90cent nodes for node i
def parse_file(filename, x, topo, percent_unit):
    latency = []
    num_pub, pubs = parse_topo(topo)
    with open(filename) as f:
        node_i = 0
        for line in f:
            tokens = line.split()   
            node_lat = {}
            for i in range(len(tokens)):
                node_lat[i] = float(tokens[i])
                if node_lat[i] > 1e5:
                    print(filename, 'has lat too large', node_lat[i])
                    print('need check')
                    sys.exit(1)

            if percent_unit == 'node':
                lat_x = get_Xcent_node(node_lat, x)
            elif percent_unit == 'pub':
                # lat_x = get_Xcent_pubs(node_lat, x, pubs)
                lat_x = get_diff_Xcent_pubs(node_i, node_lat, x, pubs, topo)
            elif percent_unit == 'hash':
                print('Not implemented. topo json file needs hash')
                sys.exit(1)
            else:
                print('Unknown percent unit', percent_unit)
                sys.exit(1)
            latency.append((node_i, lat_x))
            node_i += 1
    return latency 

def sort_lats_to_pubs(lats, pubs):
    interested_lats = [(i, lat) for i, lat in lats.items() if i in pubs]
    sorted_lats_pair = sorted(interested_lats, key=lambda item: item[1])
    return sorted_lats_pair

def sort_diff_lats_to_pub(i, lats, pubs, topo):
    loc, proc_delay = get_topo_loc_delay(topo)
    diff_lats = {}
    for m, lat in lats.items():
        if m in pubs:
            if i != m:
                line_len = (math.sqrt(
                        (loc[i][0]-loc[m][0])**2+
                        (loc[i][1]-loc[m][1])**2 ) + proc_delay[m])
                diff_lats[m] = lat - line_len
            else:
                diff_lats[i] = 0
    return sort_lats_to_pubs(diff_lats, pubs)

def parse_node_lats(stars, lats_shortest_path_file, topo, target_unit):
    latencies = {} # key is star, value is latencies
    diff_latencies = {} # key is star, value is latencies

    num_pub, pubs = parse_topo(topo)
    with open(lats_shortest_path_file) as f:
        node_i = 0
        for line in f:
            if node_i in stars:
                tokens = line.split()   
                node_lat = {}
                for i in range(len(tokens)):
                    node_lat[i] = float(tokens[i])
                    if node_lat[i] > 1e5:
                        print(filename, 'has lat too large', node_lat[i])
                        print('need check')
                        sys.exit(1)

                if target_unit == 'node':
                    print('Not implemented. topo json file needs hash')
                    sys.exit(1)
                elif target_unit == 'pub':
                    latencies[node_i] = sort_lats_to_pubs(node_lat, pubs)
                    diff_latencies[node_i] = sort_diff_lats_to_pub(node_i, node_lat, pubs, topo)
                elif target_unit == 'hash':
                    print('Not implemented. topo json file needs hash')
                    sys.exit(1)
                else:
                    print('Unknown percent unit', percent_unit)
                    sys.exit(1)
            node_i += 1
    return latencies, diff_latencies



