import sys
from plot_utility import *
import pandas as pd
import os
import seaborn as sns

if len(sys.argv) < 6:
    print('require figname<str> x_cent<int(0-100)> unit<pub/hash> exp1 exp2')
    sys.exit(1)


def plot_convergence(dirname, ax):
    epoch_file = {}
    adapts =  parse_adapt(os.path.join(dirname, 'adapts'))
    snapshots_dir = os.path.join(dirname, 'snapshots')
    index = os.path.basename(dirname).rfind('-', 0)
    exp_name = os.path.basename(dirname)[:index]
    topo_json = os.path.join(dirname, exp_name+'.json')

    for f in os.listdir(snapshots_dir):
        if 'epoch' not in f:
            continue
        e = int(f[5:-4])
        epoch_file[e] = os.path.join(snapshots_dir, f)

    sorted_file = sorted(epoch_file.items())
    epochs = [e for e, fp in sorted_file]

    data = {}
    medium_data = {}
    num_node  = -1
    for e, fp in sorted_file:
        latency = parse_file(fp, x_cent, topo_json, unit)
        data[e] = [lat for node_i, lat in latency if node_i in adapts]
        num_node = len(data[e])
        medium_data[e] = np.median(data[e])

    return data, epochs, num_node

    # df.plot(kind='box', ax=ax)

figname = sys.argv[1]
x_cent = int(sys.argv[2])
unit = sys.argv[3]
experiments = sys.argv[4:]

num_col, num_row = 1, 1
fig, ax = plt.subplots(ncols=num_col, nrows=num_row, figsize=(10*num_col,10*num_row))

colormap = plt.cm.nipy_spectral
colors = [colormap(i) for i in np.linspace(0.4, 0.9, len(experiments))]


data = {}
num_node = -1
num_exp = len(experiments)
for i, experiment in enumerate(experiments):
    latency, epochs, num_node = plot_convergence(experiment, ax)
    data[experiment] = latency


df_epoch_list = []
var_name = [str(os.path.dirname(exp)) for exp in experiments]
print(var_name)
for e in epochs:
    epoch_lat = []
    epoch_lat = np.zeros((num_node, num_exp))
    for i, exp in enumerate(experiments):
        epoch_lat[:,i] = data[exp][e]

    df = pd.DataFrame(data=epoch_lat, columns=var_name).assign(Epoch=e)
    df_epoch_list.append(df)

cdf = pd.concat(df_epoch_list)
mdf = pd.melt(cdf, id_vars=['Epoch'], var_name=['EXP'])
ax = sns.boxplot(x="Epoch", y="value", hue="EXP", data=mdf)


for i,artist in enumerate(ax.artists):
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    col = artist.get_facecolor()
    artist.set_edgecolor(col)
    artist.set_facecolor('None')

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    # Loop over them here, and use the same colour as above
    for j in range(i*6,i*6+6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

# for line in ax.get_lines()[4::12]:
    # line.set_color('white')
# for line in ax.get_lines()[10::12]:
    # line.set_color('red')
# df_plot = df.melt(id_vars='epoch', value_vars=experiments)
# sns.boxplot(x="day", y="total_bill", hue="",data=df_plot)

plt.title(str(os.path.basename(experiments[0])))
plt.tight_layout()
plt.savefig(figname)
