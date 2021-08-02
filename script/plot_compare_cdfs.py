import sys

if len(sys.argv) < 7:
    print('Require x_percent<int(0-100)/avg>, unit<node/hash>, fig-name<str>, exp1 exp2, topo<str> snapshots_dir<snapshots/snapshots-exploit> epochs... ')
    sys.exit(0)

fig_name = sys.argv[1]
out_dirs = sys.argv[2:4]
topo = sys.argv[4]
snapshot_dir = sys.argv[5]

epochs = [int(e) for e in sys.argv[6:]]
num_plot = len(out_dirs)
