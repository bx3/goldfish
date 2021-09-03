import os
import sys
import json
import mc_optimizer
import formatter
from selector import SimpleSelector
from oracle import SimpleOracle
from mc_optimizer import construct_table

# constants
table_directions = ['incoming', 'outgoing', 'bidirect']
mc_epochs = 2000 # pytorch gradient number steps
mc_lr = 1e-2     # pytorch learning rate
mc_exit_loss_diff = 1e-3 # pytorch early stop condition. See mc_optimizer.py class Stopper
top_n_peer = 2   # K NN number peer

if len(sys.argv) != 10:
    print('./goldfish.py config<str> data-path<str> curr-epoch<int> num-topo<int> output<str> num-out<int> num-in<int> num-rand<int> node-id<int>')
    print('./goldfish.py configs/node0.json stores/node0 3 2 output.json 3 6 1 3')
    sys.exit(1)

config_json = sys.argv[1]
data_path = sys.argv[2]
curr_epoch = int(sys.argv[3])
num_topo = int(sys.argv[4])
output_json = sys.argv[5]
num_out = int(sys.argv[6])
num_in = int(sys.argv[7])
num_rand = int(sys.argv[8])
node_id = int(sys.argv[9])



log_file = output_json + '.log'

# load inputs
slots = []
if curr_epoch < num_topo-1:
    print(curr_epoch, num_topo)
    sys.exit(1)
for e in range(curr_epoch-num_topo+1, curr_epoch+1):
    input_json = os.path.join(data_path, "epoch" + str(e) + "_time.json")
    if not os.path.isfile(input_json):
        sys.exit(0)

    with open(input_json) as f:
        topo_msgs = json.load(f)
        for msg in topo_msgs:
            memories = []
            if len(msg) == 0:
                print("msg epoch records needs to be non-empty")
                sys.exit(1)
            earliest = float(msg[0][1])/1e6
            for mem in msg:
                if mem[1] != "None":
                    # convert from nano sec to milli
                    memories.append([int(mem[0]), float(mem[1])/1e6-earliest, mem[2]])
                else:
                    memories.append([int(mem[0]), None, mem[2]])
            slots.append(memories)
# print(len(slots))
# for slot in slots:
    # print(slot)
    # for p,t,d in slot:
        # print(p,t,d)

# get curr outs
curr_out = set()
curr_json = os.path.join(data_path, "epoch" + str(curr_epoch) + "_time.json")
with open(curr_json) as f:
    topo_msgs = json.load(f)
    for msg in topo_msgs:
        for mem in msg:
            if mem[2] == 'bidirect' or mem[2] == 'outgoing':
                curr_out.add(int(mem[0]))

curr_out = sorted(list(curr_out))

# load config
with open(config_json) as f:
    config = json.load(f)

known_peers = []
for peer in config['peers']:
    known_peers.append(peer["id"])

# construct matrix, then complete table
mcos = mc_optimizer.McOptimizer(node_id, mc_epochs, mc_lr, mc_exit_loss_diff, top_n_peer, log_file)
selector = SimpleSelector(node_id, known_peers, num_out, num_in, None, num_rand, log_file)
oracle = SimpleOracle(1000, 1000, 1000) # invalidate oracle checks

incomplete_table, M, nM, max_time, ids, ids_direct = construct_table(slots, 0, table_directions)
topo_directions = formatter.get_topo_direction(slots, table_directions, num_topo)
            
completed_table, C, unkn_plus_mask, unkn_unab_mask = mcos.run(incomplete_table, M, 1-nM, max_time)

exploits, explores = selector.run_selector(completed_table, ids, unkn_plus_mask, nM, unkn_unab_mask, oracle, curr_out)


with open(output_json, 'w') as w:
    output = {}
    output['explores'] = explores
    output['exploits'] = exploits
    json.dump(output, w, indent=4)
