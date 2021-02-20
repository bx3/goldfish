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

# def print_mat(A):
    # for i in range(A.shape[0]):
        # text = ["{:4d}".format(int(a)) for a in A[i]]
        # line = ' '.join(text)
        # print('[' + line + ' ]')

def print_mat(A, is_float):
    for i in range(A.shape[0]):
        if not is_float:
            text = ["{:4d}".format(int(a)) for a in A[i]]
        else:
            text = ["{:5.2f}".format(a) for a in A[i]]
        line = ' '.join(text)
        print('[' + line + ' ]')
def print_two_mat(A, B, is_float):
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



class MF_tester:
    def __init__(self, T, N, L, num, out_name, add_new_data_type, num_mask_per_row, method, init_method):
        self.T = T
        self.N = N
        self.L = L
        self.num_repeat = num
        self.H_mean = 100
        self.rho_H = 0.00001 
        self.rho_W = 0
        self.filepath = 'analysis/tester/' + out_name
        self.num_alt = 5000
        self.tol_obj = 0.001
        self.identity_map = {i: i for i in range(L)}
        self.init_method = init_method
        
        self.X_add_type = add_new_data_type
        self.num_mask_per_row = num_mask_per_row

        self.inter_lat = 33 
        self.H_ref = self.construct_H(method)
        print('T', self.T, 'N', self.N, 'L', self.L, 
                'inter latency', self.inter_lat, 'H mean', np.mean(self.H_ref),
                'mask per row', self.num_mask_per_row)


    def start_mf_online(self, max_iter, num_append, std):
        X, X_noise, W_ref = self.construct_data(std)
        H_ref = self.H_ref
        W_est, H_est = None, None
        W_scores = []
        H_scores = []
        print('ref H')
        print_mat(H_ref, False)
        print()

        X_input, mask_input = self.random_mask(self.num_mask_per_row, X_noise)

        if self.init_method == 'algo':
            W_est, H_est = self.single_mf(X_input, mask_input, None, None, True)
        elif self.init_method == 'ref':
            W_est = W_ref.copy() 
            H_est = H_ref.copy() + np.random.normal(0, 0.0001, size=(self.L, self.N))

        H_mean = H_est.copy()
        W_input = W_est
        H_input = H_est

        prev_map = {}

        W_score, H_score, _, H_re, match_map = self.match_WH(W_est, H_est, W_ref, H_ref) 
        prev_map = match_map

        for i in range(0, max_iter):
            W_est, H_est = self.single_mf(X_input, mask_input, W_input, H_input, False)
            H_mean, sample_mean = self.update_H_mean(W_est, X_input, mask_input)
            
            W_score, H_score, _, H_re, match_map = self.match_WH(W_est, H_est, W_ref, H_ref) 
            if prev_map != match_map:
                print("\033[93m" + 'map change' + "\033[0m")
            prev_map = match_map
            
            W_scores.append(W_score)
            H_scores.append(H_score)
            
            # reorder correct match
            print('\t\t\t ********************************************************')
            print('iter', i, 'score', W_score, 'H_est, H_ref', W_est.shape)
            print_two_mat(H_re, H_ref, False)
            print('iter', i, 'W est')
            print_two_mat(W_est, W_ref, True)

            print('iter', i, 'X_input - W_est H_est')
            print_two_mat((X_input- W_est.dot(H_est))*mask_input, mask_input, False)
            print('iter', i, 'H_re H_ref')
            print_two_mat(H_re, H_ref, False)

            X_input, W_ref, W_input, mask_input, H_input = self.gen_new_msg(
                    self.X_add_type, X_input, mask_input,
                    W_est, W_ref, H_est, H_ref, num_append, std)
            print('iter', i, 'avg filled H_input', 'sample ean', sample_mean)
            print_mat(H_input, False)

        return W_scores, H_scores

        

    def random_mask(self, num_entry_per_row, X):
        if num_entry_per_row == 0:
            return X, np.ones(X.shape)

        num_r, num_c = X.shape
        mask = np.ones((num_r, num_c))
        all_nodes = [i for i in range(num_c)]
        for i in range(num_r):
            np.random.shuffle(all_nodes)
            missings = all_nodes[:num_entry_per_row]
            for j in missings:
                X[i,j] = 0
                mask[i,j] = 0 
        return X, mask

    def gen_new_msg(self, method, X_input, mask_input, W_est, W_ref, H_est, H_ref, num_msg, std):
        if method == 'append':
            X_input_n, W_ref, mask_input_n = self.append_X(
                    X_input, mask_input, W_ref, H_ref, num_msg, std)
        elif method == 'rolling':
            X_input_n, W_ref, mask_input_n = self.rolling_X(
                    X_input, mask_input, W_ref, H_ref, num_msg, std)
        else:
            print('Error. Unknown message batch method', method)
            sys.exit(1)

        W_input_n = self.get_new_W(W_est, H_est, X_input, num_msg)
        H_input_n = self.get_new_H(H_est, X_input, mask_input, W_est)
        return X_input_n, W_ref, W_input_n, mask_input_n, H_input_n

    def append_X(self, X, mask, W, H_ref, num_msg, std):
        new_msgs = np.zeros((num_msg, self.L))
        for i in range(num_msg):
            j = np.random.randint(self.L)
            new_msgs[i,j] = 1
        num_row =  self.T + num_msg

        W_roll = np.vstack((W, new_msgs))

        X_new = new_msgs.dot(H_ref) + np.random.normal(0, std, size=(num_msg, self.N)) 
        X_new[X_new<0] = 0
        X_new, mask_new = self.random_mask(self.num_mask_per_row, X_new)

        X_roll = np.vstack((X, X_new))
        mask = np.vstack((mask, mask_new))

        self.T += num_msg
        return X_roll, W_roll, mask

    # H_ref used for generating new rows
    def rolling_X(self, X, mask, W, H_ref, num_msg, std):
        new_msgs = np.zeros((num_msg, self.L))
        for i in range(num_msg):
            j = np.random.randint(self.L)
            new_msgs[i,j] = 1
        
        W_roll = np.zeros((self.T, self.L))
        W_roll[:self.T-num_msg] = W[num_msg:]
        W_roll[self.T-num_msg:] = new_msgs

        X_new = new_msgs.dot(H_ref) + np.random.normal(0, std, size=(num_msg, self.N)) 
        X_new[X_new<0] = 0
        X_new, mask_new = self.random_mask(self.num_entry_per_row, X_new)

        X_roll = np.zeros(X.shape)
        X_roll[:self.T-num_msg] = X[num_msg:]
        X_roll[self.T-num_msg:] = X_new

        mask_roll = np.zeros(X.shape)
        mask_roll[:self.T-num_msg] = mask[num_msg:]
        mask_roll[self.T-num_msg:] = mask_new
        return X_roll, W_roll, mask

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

    def get_new_H(self, H_est, X_input, mask, W_est):
        T = mask.shape[0]
        L = H_est.shape[0]
        N = mask.shape[1]

        H_missing = np.ones((L, N))
        for m in range(T):
            W_row = W_est[m]
            mask_row = mask[m]
            l = np.argmax(W_row)
            for j, t in enumerate(mask_row):
                if t != 0:
                    H_missing[l, j] = 0
        H_avg = np.sum(X_input * mask) / np.sum(mask)
        missings = []
        for l in range(L):
            for i in range(N):
                if H_missing[l,i] == 1:
                    missings.append(str((l,i)))
                    H_est[l,i] = H_avg

        print('set missing', ' '.join(missings))
        return H_est

    def construct_H(self, method):
        if method == 'unif':
            H = np.random.rand(self.L, self.N) * self.H_mean 
            return H
        elif method == 'log-unif':
            H = np.exp(np.random.uniform(0, 4, (self.L, self.N)))
            mean = np.mean(H)
            H = H /mean * self.H_mean/2
            # print('H', H)
            # plt.hist(H.flatten(), alpha=0.5)
            # plt.show()
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

    def add_noise(self, X, std):
        X_noise = X + np.random.normal(0, std, size=(self.T, self.N)) 
        X_noise[X_noise<0] = 0
        return X_noise

    def update_H_mean(self, W_est, X_input, mask):
        num_msg = X_input.shape[0]
        num_node = X_input.shape[1]
        num_region = W_est.shape[1]

        H_mean_data = defaultdict(list)
        sum_sample= np.sum(X_input) 
        num_sample = np.sum(mask) 
        for i in range(num_msg):
            X_row = X_input[i]
            l = np.argmax(W_est[i])
            for j, t in enumerate(X_row):
                if mask[l,j] != 0:
                    H_mean_data[(l,j)].append(t)
        sample_mean = sum_sample / num_sample
        H_mean = np.ones((num_region, num_node)) * sample_mean
        for pair, samples in H_mean_data.items():
            l,j = pair
            H_mean[l,j] = sample_mean   
        return H_mean, sample_mean

    def single_mf(self, X_input, mask, prev_W, prev_H, init_new):
        W_init, H_init = None, None
        if init_new:
            print('nndsvd')
            W_init, H_init = nndsvd.initial_nndsvd(X_input, self.L, 10)  
            # W_init, S, H_init = svds(X_input, self.L)
            # I = np.sign(W_init.sum(axis=0)) # 2 * int(A.sum(axis=0) > 0) - 1
            # W_init = W_init.dot(np.diag(I))
            # H_init = np.transpose((H_init.T).dot(np.diag(S*I)))
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
                score, W_re = self.match_WH(W_est, H_est, W, H)
                results.append(score)
            std_results.append(sum(results)/len(results))

        plt.scatter(stds, std_results)
        plt.title('mf analysis by brute forcing closest W')
        plt.xlabel('gaussian noise std. data mean is set 100. ' + 'each dot is mean of ' + str(self.num_repeat) + 'mf')
        plt.ylabel('correctness ratio')
        plt.show()

    def match_WH(self, W_est, H_est, W_ref, H_ref):
        est_to_ori, H_score = self.get_est_mapping(H_ref, H_est)
        row, col = W_est.shape
        W_re = np.zeros((row, col))
        H_re = np.zeros(H_ref.shape)
        for i in range(col):
            re_idx = est_to_ori[i]
            W_re[:, re_idx] = W_est[:, i]
            H_re[re_idx] = H_est[i, :]

        W_score = self.compare_W(W_ref, W_re, self.identity_map)
        return W_score, H_score, W_re, H_re, est_to_ori

    def compare_W(self, W_ref, W_est, est_to_ori):
        W_re = np.zeros(W_est.shape)
        row, col = W_est.shape
        for i in range(col):
            re_idx = est_to_ori[i]
            W_re[:, re_idx] = W_est[:, i]

        results = np.argmax(W_ref, axis=1) == np.argmax(W_re, axis=1)
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

