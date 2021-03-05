import numpy as np
import solver
import nndsvd
from scipy.sparse.linalg import svds
from scipy.linalg import eigh
import sys
from numpy.linalg import inv
from sklearn.decomposition import NMF
from numpy import linalg as LA
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from bandit import Bandit

class MF_tester:
    def __init__(self, T, N, L, num, out_name, add_new_data_type, num_mask_per_row, method, init_method, mask_method):
        self.T = T
        self.N = N
        self.L = L
        self.num_repeat = num
        self.H_mean = 100
        self.rho_H = 0
        self.rho_W = 0
        self.fig_filepath = 'analysis/tester/' + out_name
        self.result_filepath = 'analysis/tester_result/' + out_name
        self.num_alt = 5000
        self.tol_obj = 0.00001

        self.init_method = init_method
        
        self.X_add_type = add_new_data_type
        self.num_mask_per_row = num_mask_per_row

        self.inter_lat = 33
        self.H_ref = self.construct_H(method)
        print('T', self.T, 'N', self.N, 'L', self.L, 
                'inter latency', self.inter_lat, 'H mean', np.mean(self.H_ref),
                'mask per row', self.num_mask_per_row)

        self.mask_method = mask_method
        self.id = 0
        self.bandit_alpha = 2

        self.bandit = Bandit(
            self.id,
            self.L,
            self.N,
            self.bandit_alpha,
            'lcb'
        )
        
    def start_mf_online(self, max_iter, num_append, std):
        X, X_noise, W_ref = self.construct_data(std)
        H_ref = self.H_ref
        # W_ref, H_ref = self.augment_dim(W_ref, H_ref)

        W_est, H_est, W_prev, H_prev = None, None, None, None
        W_scores = []
        H_scores = []
        print('ref H')
        print_mat(H_ref, False)
        print()
        
        X_input, mask_input = self.get_mask_input(self.num_mask_per_row, X_noise, True)

        if self.init_method == 'algo':
            W_est, H_est = self.single_mf(X_input, mask_input, None, None, True)
        elif self.init_method == 'ref':
            W_est = W_ref.copy() 
            H_est = H_ref.copy() + np.random.normal(0, 0.0001, size=(self.L, self.N))

        W_prev = W_est.copy() 
        H_prev = H_est.copy()

        H_mean = H_est.copy()
        W_input = W_est
        H_input = H_est

        prev_map = {}

        W_score, H_score, _, _, match_map = self.match_WH(W_est, H_est, W_ref, H_ref, True) 
        prev_map = match_map

        for i in range(0, max_iter):
            print('\t\t\t ******************************************************** niter '+
                    str(i) + ' ********************************************************' )
            W_est, H_est = self.single_mf(X_input, mask_input, W_input, H_input, False)
            # print('W_est')
            # print_mat(W_est, True)
            # print('W_prev')
            # print_mat(W_prev, True)
            # print('H_est')
            # print_mat(H_est, False)
            # print('H_prev')
            # print_mat(H_prev, False)

            _, _, W_reorder, H_reorder, _ = self.match_WH(W_est, H_est, W_prev, H_prev, False) 
            W_prev = W_reorder.copy()
            H_prev = H_reorder.copy()

            W_score, H_score, W_reorder_ref, H_reorder_ref, match_map = self.match_WH(W_est, H_est, W_ref, H_ref, True) 

            if prev_map != match_map:
                print("\033[93m" + 'map change' + "\033[0m")
            prev_map = match_map

            self.update_bandit(W_reorder, H_reorder, X_input, mask_input)
            H_mean, sample_mean, H_mean_mask = self.get_H_mean(W_reorder, X_input, mask_input)
            
            W_scores.append(W_score)
            H_scores.append(H_score)
            
            # reorder correct match
            
            # print('iter', i, 'score', W_score, 'H_est, H_ref', W_est.shape)
            # print_two_mat(H_reorder, H_ref, False)
            # print('iter', i, 'W est')
            # print_three_mat(W_est, W_ref, self.get_argmax_W(W_est), [True,False,False])
            print('iter',i, 'W_est', 'W_reorder_ref', 'W_ref', 'result', 'masked_X', 'estToOri', match_map)
            print_mats([format_mat(W_reorder,True), 
                        format_mat(self.get_argmax_W(W_reorder_ref), False),
                        format_mat(W_ref,False), 
                        format_array(np.argmax(W_reorder_ref, axis=1) == np.argmax(W_ref, axis=1)),
                        format_masked_mat(X_input, mask_input, False)])

            # print_mats([format_mat(W_reorder,True), format_mat(W_ref,False),
                        # format_mat(self.get_argmax_W(W_reorder), False),format_array(result),
                        # format_masked_mat(X_input-W_reorder.dot(H_reorder), mask_input, False)])

            print('iter', i, 'H_reorder H_ref')
            print_mats([format_masked_mat(H_reorder, H_mean_mask, False), 
                format_masked_mat(H_ref, H_mean_mask, False)])
            # print_two_mat(H_est, H_ref, False)
            # print('iter', i, 'masked H_mean', 'mean', sample_mean)
            # print_mats([format_masked_mat(H_mean, H_mean_mask, False)])
            # print_masked_mat(H_mean, H_mean_mask, False)
            X_input, W_ref, W_input, mask_input, H_input = self.gen_new_msg(
                    self.X_add_type, X_input, mask_input,
                    W_reorder, W_ref, 
                    H_reorder, H_mean, H_mean_mask, H_ref, num_append, std)

            print('iter', i, 'H_input', 'H_reorder_ref', 'num miss', np.sum(H_mean_mask!=1))
            print_mats([format_mat(H_input, False), 
                format_mat(H_reorder_ref, False)])

        return W_scores, H_scores

    def update_bandit(self, W_est, H_est, X_input, mask_input):
        max_time = np.max(X_input)
        self.bandit.set_ucb_table(W_est, X_input, mask_input, max_time)

    def get_mask_input(self, num_mask_per_row, X, is_init):
        if num_mask_per_row == 0:
            return X, np.ones(X.shape)

        if self.mask_method == 'random' or is_init:
            num_r, num_c = X.shape
            mask = np.ones((num_r, num_c))
            all_nodes = [i for i in range(num_c)]
            all_nodes.remove(self.id)
            for i in range(num_r):
                np.random.shuffle(all_nodes)
                missings = all_nodes[:num_mask_per_row-1]
                for j in missings:
                    X[i,j] = 0
                    mask[i,j] = 0 
                    mask[i, self.id] = 0
            return X, mask
        elif self.mask_method == 'bandit':
            valid_arms = [i for i in range(self.N)]
            valid_arms.remove(self.id)
            num_arm = self.N - num_mask_per_row
            scores, num_samples, max_time, bandit_T,score_mask = self.bandit.get_scores()
            regions_arms = self.bandit.pull_arms(valid_arms, num_arm)
            arms = [a for l, a in regions_arms]
        
            # print('Score. max time.', max_time, 'T', bandit_T )
            # print_mats([format_masked_mat(scores, score_mask, True), format_mat_5(num_samples, False)])
            # print('pulled arms:', regions_arms, sorted(arms))

            num_r, num_c = X.shape
            mask = np.zeros((num_r, num_c))
            for a in arms:
                mask[:,a] = 1
            X = X * mask
            return X, mask
        else:
            print('Error. Unknown mask method', self.mask_method)
            sys.exit(1)

    def gen_new_msg(self, method, X_input, mask_input, W_est, W_ref, H_est, H_mean, H_mean_mask, H_ref, num_msg, std):
        if method == 'append':
            X_input_n, W_ref, mask_input_n = self.append_data(
                    X_input, mask_input, W_ref, H_ref, num_msg, std)
        elif method == 'rolling':
            X_input_n, W_ref, mask_input_n = self.rolling_data(
                    X_input, mask_input, W_ref, H_ref, num_msg, std)
        else:
            print('Error. Unknown message batch method', method)
            sys.exit(1)
        # used for mf initialization
        W_input_n = self.get_new_W(W_est, H_est, X_input, num_msg)
        H_input_n = self.get_new_H(H_est, H_mean, H_mean_mask, X_input, mask_input, W_est)

        return X_input_n, W_ref, W_input_n, mask_input_n, H_input_n

    # append rows to X, W, mask
    def append_data(self, X, mask, W, H_ref, num_msg, std):
        new_msgs = np.zeros((num_msg, self.L))
        for i in range(num_msg):
            j = np.random.randint(self.L)
            new_msgs[i,j] = 1
        num_row =  self.T + num_msg
        W_append = np.vstack((W, new_msgs))

        X_new = new_msgs.dot(H_ref) + np.random.normal(0, std, size=(num_msg, self.N)) 
        X_new[X_new<0] = 0
        X_new, mask_new = self.get_mask_input(self.num_mask_per_row, X_new, False)

        X_append = np.vstack((X, X_new))
        mask = np.vstack((mask, mask_new))

        self.T += num_msg
        return X_append, W_append, mask

    # H_ref used for generating new rows
    def rolling_data(self, X, mask, W, H_ref, num_msg, std):
        print('rolling data', num_msg, self.T)
        new_msgs = np.zeros((num_msg, self.L))
        for i in range(num_msg):
            j = np.random.randint(self.L)
            new_msgs[i,j] = 1
        
        W_roll = np.zeros((self.T, self.L))
        W_roll[:self.T-num_msg] = W[num_msg:]
        W_roll[self.T-num_msg:] = new_msgs

        X_new = new_msgs.dot(H_ref) + np.random.normal(0, std, size=(num_msg, self.N)) 
        X_new[X_new<0] = 0
        X_new, mask_new = self.get_mask_input(self.num_mask_per_row, X_new, False)

        X_roll = np.zeros(X.shape)
        X_roll[:self.T-num_msg] = X[num_msg:]
        X_roll[self.T-num_msg:] = X_new

        mask_roll = np.zeros(X.shape)
        mask_roll[:self.T-num_msg] = mask[num_msg:]
        mask_roll[self.T-num_msg:] = mask_new
        return X_roll, W_roll, mask_roll

    def get_new_W(self, W_est, H_est, X, num_msg):
        if self.X_add_type == 'rolling':
            W_out = np.zeros(W_est.shape)
            new_noise = np.random.rand(num_msg, W_est.shape[1])
            W_out[:self.T-num_msg] = W_est[num_msg:]
            W_out[self.T-num_msg:] = new_noise
        elif self.X_add_type == 'append':
            new_noise = np.random.rand(num_msg, W_est.shape[1])
            W_out = np.vstack((W_est, new_noise))
        return W_out

    def get_new_H(self, H_est, H_mean, H_mean_mask, X_input, mask, W_est):
        T = mask.shape[0]
        L = H_est.shape[0]
        N = mask.shape[1]

        # H_missing = np.ones((L, N))
        # for m in range(T):
            # W_row = W_est[m]
            # mask_row = mask[m]
            # l = np.argmax(W_row)
            # for j, t in enumerate(mask_row):
                # if t != 0:
                    # H_missing[l, j] = 0

        # H_avg = np.sum(X_input * mask) / np.sum(mask)
        # missings = []
        # missings_str = []
        # for l in range(L):
            # for i in range(N):
                # if H_missing[l,i] == 1:
                    # if i != self.id:
                        # missings_str.append(str((l,i)))
                        # missings.append((l,i))
                    # H_est[l,i] = 0 # H_avg

        # for l,i in missings:
            # assert(H_mean_mask[l,i] == 0)
        mean =  np.sum(H_mean*H_mean_mask)/np.sum(H_mean_mask)
        H_new = H_mean*H_mean_mask + mean*(H_mean_mask!=1)
        return H_new 

    def construct_H(self, method):
        if method == 'unif':
            H = np.random.rand(self.L, self.N) * self.inter_lat * 2 
            return H
        elif method == 'log-unif':
            H = np.exp(np.random.uniform(0, 4, (self.L, self.N)))
            mean = np.mean(H)
            H = H /mean * self.inter_lat/2
            return H
        elif method == '1D-linear':
            return self.construct_linear_H(self.inter_lat)
        elif method == 'datacenter':
            return self.construct_datacenter_H(self.inter_lat)
        else:
            print('Error. Unknown H-dist', method)
            sys.exit(1)

    def convert_1D_time(self, r1, r2, inter_lat):
        return abs(r1-r2) * inter_lat

    def construct_linear_H(self, inter_lat):
        H = np.zeros((self.L, self.N))
        node_region = {}
        num_nodes_per_region = self.N / self.L
        for i in range(self.N):
            node_region[i] = int(i/num_nodes_per_region)
        
        for l in range(self.L):
            for i in range(self.N):
                H[l, i] = self.convert_1D_time(l, node_region[i], inter_lat)

        mean_H = np.mean(H)
        H = H * 50.0/ mean_H
        print("1D linear H")
        print_mat(H, False)
        return H

    def construct_datacenter_H(self, inter_lat):
        H = np.zeros((self.L, self.N)) 
        node_region = {}
        num_node_per_region = self.N / self.L
        for i in range(self.N):
            node_region[i] = int(i / num_node_per_region)
        for l in range(self.L):
            for i in range(self.N):
                if l == node_region[i]:
                    H[l, i] = 0 
                else:
                    H[l, i] = inter_lat
        print("datacenter H")
        print_mat(H)
        return H

    # std is noise standard deviation
    def construct_data(self, std):
        W = np.zeros((self.T, self.L))
        for i in range(self.T):
            j = np.random.randint(self.L)
            W[i,j] = 1.0
        X = W.dot(self.H_ref)

        X_noised = X + np.random.normal(0, std, size=(self.T, self.N)) 
        X_noised[X_noised<0] = 0
        return X, X_noised, W

    def augment_dim(self, W, H):
        T = W.shape[0]
        w = np.random.uniform(0, 1, size=(T, 1))
        W = np.hstack((W,w))
        
        # h = np.zeros((1, self.N))
        h = np.random.uniform(0, self.inter_lat, (1, self.N))
        H = np.vstack((H, h))
        self.L += 1
        return W, H

    def add_noise(self, X, std):
        X_noise = X + np.random.normal(0, std, size=(self.T, self.N)) 
        X_noise[X_noise<0] = 0
        return X_noise

    def get_H_mean(self, W_est, X_input, mask):
        T = X_input.shape[0]
        N = X_input.shape[1]
        L = W_est.shape[1]

        H_mean_data = defaultdict(list)
        sum_sample= np.sum(X_input*mask) 
        num_sample = np.sum(mask) 
        for i in range(T):
            X_row = X_input[i]
            l = np.argmax(W_est[i])
            for j, t in enumerate(X_row):
                if mask[i,j] != 0:
                    H_mean_data[(l,j)].append(t)
        sample_mean = sum_sample / num_sample
        H_mean = np.ones((L, N)) * sample_mean
        H_mean_mask = np.zeros((L, N))
        # for k, v in sorted(H_mean_data.items()):
            # print(k,np.round(v))
        for pair, samples in H_mean_data.items():
            l,j = pair
            H_mean[l,j] = sum(samples)/len(samples)   
            H_mean_mask[l,j] = 1
        return H_mean, sample_mean, H_mean_mask

    def row_wise_div(self, H, a):
        for i in range(H.shape[0]):
            H[i] = H[i]/a[i]
        return H

    def single_mf(self, X_input, mask, prev_W, prev_H, init_new):
        W_init, H_init = None, None
        if init_new:
            # W_init, H_init = nndsvd.initial_nndsvd(X_input, self.L, 10)  

            # W_init_row_sum = np.max(W_init, axis=1)
            # H_init_row_sum = np.max(H_init, axis=1)

            # W_init = self.row_wise_div(W_init, W_init_row_sum)
            # H_init = self.row_wise_div(H_init, H_init_row_sum)
            # W_init, S, H_init = svds(X_input, self.L)
            X_mean = np.sum(X_input*mask) / np.sum(mask)
            W_init = np.random.uniform(0, 1, (self.T, self.L))
            H_init = np.random.uniform(0, X_mean, (self.L, self.N))

            print('random W_init')
            print_mat(W_init, True)
            print('random H_init')
            print_mat(H_init, True)
        else:
            W_init, H_init = prev_W, prev_H
    
        W_est, H_est, opt = solver.alternate_minimize(
                W_init, H_init, X_input, mask,
                self.L, None, 0, 
                self.num_alt, self.tol_obj, self.rho_W, self.rho_H)
        print('opt', opt)
        return W_est, H_est

    def test_mf(self):
        results = []
        stds = [i*10 for i in range(int(self.H_mean/10))]
        std_results = []
        for std in stds:
            results = []
            for _ in range(self.num_repeat):
                X, W, H = self.construct_data()
                X_noise = self.add_noise(X, std)
                W_est, H_est = self.single_mf(X_noise, W, H, True)
                score, W_re = self.match_WH(W_est, H_est, W, H, True)
                results.append(score)
            std_results.append(sum(results)/len(results))

        plt.scatter(stds, std_results)
        plt.title('mf analysis by brute forcing closest W')
        plt.xlabel('gaussian noise std. data mean is set 100. ' + 'each dot is mean of ' + str(self.num_repeat) + 'mf')
        plt.ylabel('correctness ratio')
        plt.show()


    def match_WH(self, W_est, H_est, W_ref, H_ref, is_compare):
        est_to_ori, H_score = self.get_est_mapping(H_ref, H_est)
        row, col = W_est.shape
        W_re = np.zeros((row, col))
        H_re = np.zeros(H_ref.shape)
        for i in range(col):
            re_idx = est_to_ori[i]
            W_re[:, re_idx] = W_est[:, i]
            H_re[re_idx] = H_est[i, :]
        W_score = -1
        if is_compare:
            W_score = self.compare_W(W_ref, W_re, {i: i for i in range(self.L)})
        return W_score, H_score, W_re, H_re, est_to_ori

    def compare_W(self, W_ref, W_est, est_to_ori):
        W_re = np.zeros(W_est.shape)
        row, col = W_est.shape
        for i in range(col):
            re_idx = est_to_ori[i]
            W_re[:, re_idx] = W_est[:, i]

        a = np.argmax(W_ref, axis=1)
        b = np.argmax(W_re, axis=1)
        assert(len(a) == len(b))
        results = a == b
        W_score = sum(results)/len(results)
        return W_score
    

    def best_perm(self, table):
        num_node = table.shape[0]
        nodes = [i for i in range(num_node)]
        best_comb = None
        best = 999999
        hist = {}
        for comb in itertools.permutations(nodes, num_node):
            score = 0
            # i is ori index, j in estimate index
            for i in range(num_node):
                j = comb[i]
                score += table[i,j]  
            hist[comb] = score
            if best_comb is None or score < best:
                best_comb = comb
                best = score
        est_to_ori = {}
        for i in range(num_node):
            est = best_comb[i]
            est_to_ori[est] = i

        return est_to_ori, best 


    def get_est_mapping(self, H, H_est):
        table = np.zeros((len(H), len(H_est)))
        est_to_ori = {}
        for i in range(len(H)):
            u = H[i,:]
            for j in range(len(H_est)):
                v = H_est[j,:]
                dis = LA.norm(u-v) 
                table[i,j] = dis
        est_to_ori, H_score = self.best_perm(table)

        # for i in range(len(H)):
            # est = np.argmin(table[i,:])
            # est_to_ori[est] = i 
        # print(m)
        # print(est_to_ori)
        return est_to_ori, H_score

    def svd_init(self, X):
        A, S, B = svds(X, self.L)
        I = np.sign(A.sum(axis=0)) # 2 * int(A.sum(axis=0) > 0) - 1
        A = A.dot(np.diag(I))
        B = np.transpose((B.T).dot(np.diag(S*I)))
        return A, B

    def uniform_init(self, X):
        W_init = np.random.rand(self.T, self.L)
        H_init = np.random.rand(self.L, self.N) * self.H_mean 
        return W_init, H_init


    def get_argmax_W(self, W_est):
        chosen = np.argmax(W_est, axis=1)
        W_chosen = np.zeros(W_est.shape)
        for i, c in enumerate(chosen):
            W_chosen[i,c] = 1
        return W_chosen

    def debug(W, H, W_est, H_est, X):
        print('W')
        print_mat(W)
        print('H')
        print_mat(H)   
        print('X')
        print_mat(X)   
        print('W estimate')
        print_mat(W_est)
        print('H estimate')
        print_mat(H_est)   

    def square_root(self, X):
        p = X.T.dot(X)

        # model = NMF(n_components=L, init='nndsvd', random_state=0)
        # A = model.fit_transform(p)
        # H = model.components_
        # W, H = nndsvd.initial_nndsvd(p, L, 1) #svd_init(X, L)  # 

        w, v = eigh(p, subset_by_index=[self.N-self.L, self.N-1])
        # print(w/max(w)*T)
        # sys.exit(2) 
        # print_matrix(A.T)
        # print()

        # print_matrix(H)
        # print()
        # te = inv(H.dot(H.T))
        # inverse_H = H.T.dot(te)
        # W = X.dot(inverse_H)
        # print_matrix(W)


        #w, v = eigh(p, subset_by_index=[N-L, N-1]) #, subset_by_index=[N-L, N-1]) # largest L eigenvalue
        # print(v)
        # print(w)

        # print_matrix(H_init)
        # c = H_init.T.dot(H_init)
        # print_matrix(X)
        # print_matrix(H_init)

        # print('W')
        # print_matrix(W_init)
        # sys.exit(2)
        return W, H

        
            # if  == 'rolling':
                # X_input, W_ref = self.rolling_X(X_input, W_ref, H_est, H_ref, num_append, std)
            # elif self.X_add_type == 'append':
                # X_input, W_ref = self.append_X(X_input, W_ref, H_ref, num_append, std)
            # W_input = self.get_new_W(W_est, H_est, X_input, num_append)

# def print_mat(A):
    # for i in range(A.shape[0]):
        # text = ["{:4d}".format(int(a)) for a in A[i]]
        # line = ' '.join(text)
        # print('[' + line + ' ]')

def format_mat(A, is_float):
    lines = []
    if not is_float:
        ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
    else:
        ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]
    lines.append('_' + '_'.join(ticks)+'_ ')
    for i in range(A.shape[0]):
        if not is_float:
            text = ["{:4d}".format(int(a)) for a in A[i]]
        else:
            text = ["{:5.2f}".format(a) for a in A[i]]
        line = ' '.join(text)
        lines.append('[' + line + ' ]')
    return lines

def format_array(a):
    lines = []
    title = '_Rslt'
    lines.append('_' + ''.join(title)+'_ ')
    for i in range(len(a)):
        text = ["{:5s}".format(str(a[i]))]
        line = ' '.join(text)
        lines.append(' ' + line + '  ')
    return lines


def format_mat_5(A, is_float):
    lines = []
    if not is_float:
        ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]
    else:
        ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]
    lines.append('_' + '_'.join(ticks)+'_ ')
    for i in range(A.shape[0]):
        if not is_float:
            text = ["{:5d}".format(int(a)) for a in A[i]]
        else:
            text = ["{:5.2f}".format(a) for a in A[i]]
        line = ' '.join(text)
        lines.append('[' + line + ' ]')
    return lines


def print_mats(mats):
    lines = []
    num_mat = len(mats)
    num_line = len(mats[0])
    for i in range(num_line): 
        line = ''
        for mat in mats:
            line += mat[i] + '\t'
        print(line) 

def print_mat(A, is_float):
    for i in range(A.shape[0]):
        if not is_float:
            text = ["{:4d}".format(int(a)) for a in A[i]]
        else:
            text = ["{:5.2f}".format(a) for a in A[i]]
        line = ' '.join(text)
        print('[' + line + ' ]')

def format_masked_mat(A, mask, is_float):
    lines = []
    if not is_float:
        ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
    else:
        ticks = [str(a).rjust(5, '_') for a in range(A.shape[1])]

    lines.append('_' + '_'.join(ticks)+'_ ')
    for i in range(A.shape[0]):
        row = A[i]
        text = []
        if not is_float:
            for j in range(len(row)):
                if mask[i,j] == 1:
                    text.append("{:4d}".format(int(row[j]))) 
                elif mask[i,j] == 0:
                    text.append("{:>4}".format('*')) 
        else:
            for j in range(len(row)):
                if mask[i,j] == 1:
                    text.append("{:5.2f}".format(row[j])) 
                elif mask[i,j] == 0:
                    text.append("{:>5}".format('*')) 
        line = ' '.join(text)
        lines.append('[' + line + ' ]')
    return lines

def print_masked_mat(A, mask, is_float):
    ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
    print('_' + '_'.join(ticks))
    for i in range(A.shape[0]):
        row = A[i]
        text = []
        if not is_float:
            for j in range(len(row)):
                if mask[i,j] == 1:
                    text.append("{:4d}".format(int(row[j]))) 
                elif mask[i,j] == 0:
                    text.append("{:>4}".format('*')) 
        else:
            for j in range(len(row)):
                if mask[i,j] == 1:
                    text.append("{:5.2f}".format(row[j])) 
                elif mask[i,j] == 0:
                    text.append("{:>5}".format('*')) 
        line = ' '.join(text)
        print('[' + line + ' ]')

# def format_two_mat(A, B, is_float):
    # lines = []
    # ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
    # lines.append(ticks+'_ ')
    # for i in range(A.shape[0]):
        # if is_float:
            # text = ["{:5.2f}".format(a) for a in A[i]]
        # else:
            # text = ["{:4d}".format(int(a)) for a in A[i]]

        # line = ' '.join(text)
        # if is_float:
            # text = ["{:5.2f}".format(a) for a in B[i]]
        # else:
            # text = ["{:4d}".format(int(a)) for a in B[i]]

        # line2 = ' '.join(text)
        # print('[' + line + ' ] \t\t' + '[' + line2 + ' ]')
        # lines.append()

def print_two_mat(A, B, is_float):
    ticks = [str(a).rjust(4, '_') for a in range(A.shape[1])]
    print('_' + '_'.join(ticks))
    for i in range(A.shape[0]):
        if is_float:
            text = ["{:5.2f}".format(a) for a in A[i]]
        else:
            text = ["{:4d}".format(int(a)) for a in A[i]]

        line = ' '.join(text)
        if is_float:
            text = ["{:5.2f}".format(a) for a in B[i]]
        else:
            text = ["{:4d}".format(int(a)) for a in B[i]]

        line2 = ' '.join(text)
        print('[' + line + ' ] \t\t' + '[' + line2 + ' ]')

def print_three_mat(A, B, C, is_float):
    for i in range(A.shape[0]):
        if is_float[0]:
            text = ["{:5.2f}".format(a) for a in A[i]]
        else:
            text = ["{:4d}".format(int(a)) for a in A[i]]

        line = ' '.join(text)
        if is_float[1]:
            text = ["{:5.2f}".format(a) for a in B[i]]
        else:
            text = ["{:4d}".format(int(a)) for a in B[i]]

        line2 = ' '.join(text)

        if is_float[2]:
            text = ["{:5.2f}".format(a) for a in C[i]]
        else:
            text = ["{:4d}".format(int(a)) for a in C[i]]

        line3 = ' '.join(text)

        print('[' + line + ' ] \t\t' + '[' + line2 + ' ] \t\t'+ '[' + line3 + ' ]')

def print_four_mat_mask(A, B, C, is_float, D, mask, answer):
    for i in range(A.shape[0]):
        if is_float[0]:
            text = ["{:5.2f}".format(a) for a in A[i]]
        else:
            text = ["{:4d}".format(int(a)) for a in A[i]]

        line = ' '.join(text)
        if is_float[1]:
            text = ["{:5.2f}".format(a) for a in B[i]]
        else:
            text = ["{:4d}".format(int(a)) for a in B[i]]

        line2 = ' '.join(text)

        if is_float[2]:
            text = ["{:5.2f}".format(a) for a in C[i]]
        else:
            text = ["{:4d}".format(int(a)) for a in C[i]]

        line3 = ' '.join(text)

        row = D[i]
        text = []
        for j in range(len(row)):
            if mask[i,j] == 1:
                text.append("{:4d}".format(int(row[j]))) 
            elif mask[i,j] == 0:
                text.append("{:>4}".format('*'))
        line4 = ' '.join(text)

        print('[' + line + ' ] \t\t' + '[' + line2 + ' ] \t\t'+ '[' + line3 + ' ] \t\t' + 
                str(answer[i]) + '\t' + '[' + line4 + ' ]')


