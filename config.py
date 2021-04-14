# reduced link  
use_reduce_link = False 
reduce_link_ratio = 0.1

# network config
input_json = 'inputs/rand_nodes_40.json'


link_file = "inputs/datacenter4_nodes7_inter200_intra20_rand.txt"
# datacenter4_nodes7_inter200_intra20_rand.txt
# datacenter4_nodes5_rand_inter200_intra20
# datacenter4_nodes5_rand_inter200_intra20.txt
# datacenter4_nodes6_inter200_intra20_rand.txt
# datacenter4_nodes10_inter200_intra20_rand.txt
num_dc_region = 4 # including single isolated node
# datacenter4_nodes5_rand_inter200_intra20.txt
# datacenter3_nodes5_inter200_intra10.txt
# datacenter4_nodes10_inter200_intra5.txt
# inputs/datacenter5_nodes8_inter200_intra5.txt
# datacenter5_nodes20_inter200_intra50.txt
# inputs/weight1.txt
# datacenter5_nodes20_inter200_intra50.txt
# weight1.txt datacenter5_nodes4_inter200_intra50.txt
num_node = 40
in_lim = num_node # 40 

# load conns state from prev experiment 
is_load_conn = False 
conn_path = 'inputs/conn.txt'
num_thread = 1 #min(num_node, 60)

is_elimiate_lucky_W = False 
eliminate_threshold = 0.2

# neigbor selection method
use_matrix_completion = True 
use_2hop = False 

# optimizer
window_constant = 2 
num_msg = 6
use_abs_time = True 
tol_obj = 0.00001    # exit optimization difference
num_alt = 5000  # max exit alteration step
max_step = 1  # within W H how many step
# initialization
init_nndsvd = True # use of nndsvd encourage sparsity for both W and H 
nndsvd_seed = None # if use nndsvd, seed can be set for consistency
# feedback
feedback_WH = True 
prior_WH = True 
W_noise_mean = 0.1
W_noise_std = 0.1
H_noise_std_ratio = 4
num_batch = 4

rho_H = 0.1
rho_W = 0

# bandit
alpha = 0.2
time_constant = 999
hard_update = True   # otherwise soft 
ucb_method = 'lcb'
num_untouch_arm = 0


# node config
num_keep = 2
num_2_hop = 2
num_3_hop = 0
num_random = 1

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

data_file = "inputs/data1.txt"
data_dir = "data"
node_delay_mean = 0
node_delay_std =0 


# broadcast detail
MISMATCH = 0.00001
