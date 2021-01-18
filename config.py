# reduced link  
use_reduce_link = False 
reduce_link_ratio = 0.1

# load conns state from prev experiment 
is_load_conn = False 
conn_path = 'inputs/conn.txt'
num_thread = 10 # divide num nodes, so works are spread evenly

# network config
num_node = 200
in_lim = 200 # 40 
num_batch = 4  # one batch contains msg of window / num_batch

# neigbor selection method
use_matrix_completion = True 
use_2hop = False 

# optimizer
window_constant = 1
use_abs_time = True 
tol_obj = 0.01    # exit optimization difference
num_alt = 5000  # max exit alteration step
max_step = 1  # within W H how many step
init_nndsvd = True # use of nndsvd encourage sparsity for both W and H 
nndsvd_seed = None # if use nndsvd, seed can be set for consistency
feedback_WH = False 

# bandit
alpha = 2 
time_constant = 2500 
hard_update = True   # otherwise soft 
is_ucb = True # ucb: argmax (c - t) / c, lcb: argmin t


# node config
num_keep = 3
num_2_hop = 3
num_3_hop = 0
num_random = 2

# how to choose 1 hops
both_in_and_out = True 

# how to choose two,three hops
is_sort_score = False   
is_favor_new = False 
is_per_recommeder_select = True 
is_rank_occurance = False 

# history decay
use_score_decay = False 
old_weight = 0.5 
new_weight = 1 - old_weight




# attack
num_adv = 0             # out of num node
adv_hash = 0.1          # percentage adversarial power
worst_conn_attack = False 
recommend_worst_attack = False 
sybil_update_priority = False 

# peers info. If dynamic, some peers in the loop may already 
# have num_keep+ peers, while early node only knows num_keep 
# peers from other nodes.
is_dynamic = False #True



network_type = 'unhash'
method = 'subset'
use_sequential = False 

# graph info
data_index = 1
hash_file = "inputs/hash1.txt"
link_file = "inputs/weight1.txt" # "datacenter5_nodes4_inter200_intra10.txt" # "weight1.txt"
data_file = "inputs/data1.txt"
data_dir = "data"

# broadcast detail
MISMATCH = 0.00001
