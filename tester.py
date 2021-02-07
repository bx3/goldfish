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

def print_mat(A):
    for i in range(A.shape[0]):
        print(list(np.round(A[i], 0)))

class MF_tester:
    def __init__(self, T, N, L, num, out_name):
        self.T = T
        self.N = N
        self.L = L
        self.num_repeat = num
        self.H_mean = 100
        self.rho_H =  0 #0.1 
        self.rho_W = 0
        self.filepath = 'analysis/tester/' + out_name
        self.num_alt = 5000
        self.tol_obj = 0.001
        self.identity_map = {i: i for i in range(L)}
        print('T', self.T, 'N', self.N, 'L', self.L)

    def construct_data(self):
        W = np.zeros((self.T, self.L))
        for i in range(self.T):
            j = np.random.randint(self.L)
            W[i,j] = 1
        H = np.random.rand(self.L, self.N) * self.H_mean 
        X = W.dot(H)
        return X, W, H

    def add_noise(self, X, std):
        X_noise = X + np.random.normal(0, std, size=(self.T, self.N)) 
        X_noise[X_noise<0] = 0
        return X_noise

    # H_ref used for generating new rows
    def rolling_X(self, X, W, H_ref, num_msg, std):
        new_msgs = np.zeros((num_msg, self.L))
        for i in range(num_msg):
            j = np.random.randint(self.L)
            new_msgs[i,j] = 1
        
        W_roll = np.zeros((self.T, self.L))
        W_roll[:self.T-num_msg] = W[num_msg:]
        W_roll[self.T-num_msg:] = new_msgs

        X_new = new_msgs.dot(H_ref) + np.random.normal(0, std, size=(num_msg, self.N)) 
        X_new[X_new<0] = 0

        X_roll = np.zeros(X.shape)
        X_roll[:self.T-num_msg] = X[num_msg:]
        X_roll[self.T-num_msg:] = X_new

        return X_roll, W_roll

    def set_W(self, W_est, H_est, X, num_msg):
        # # random
        # W_out = np.random.rand(W_est.shape[0], W_est.shape[1])

        # # partial random
        W_out = np.zeros(W_est.shape)
        new_noise = np.random.rand(num_msg, W_est.shape[1])
        W_out[:self.T-num_msg] = W_est[num_msg:]
        W_out[self.T-num_msg:] = new_noise

        # te = inv(H_est.dot(H_est.T))
        # inverse_H = H_est.T.dot(te)
        # W_out = X.dot(inverse_H)
        return W_out

    def online_test_mf(self, max_iter, num_append, std):
        X, W_ref, H_ref = self.construct_data()
        X_noise = self.add_noise(X, std)
        W_scores = []
        H_scores = []
        print('num_append', num_append)
        print('ref H')
        print_mat(H_ref)
        print()
        X_input = X_noise
        W_est, H_est = self.single_mf(X_input, None, None, True)
        W_score, H_score, _, H_re, match_map = self.match_WH(W_est, H_est, W_ref, H_ref) 
        prev_map = match_map

        W_scores.append(W_score)
        H_scores.append(H_score)
        print('iter', 0, 'W_score', W_score)
        print_mat(H_est)
        print(match_map)
        print()
        for i in range(1, max_iter):
            # correct answer
            X_input, W_ref = self.rolling_X(X_input, W_ref, H_ref, num_append, std)
            W_input = self.set_W(W_est, H_est, X_input, num_append)
            W_est, H_est = self.single_mf(X_input, W_input, H_est, False)
            W_score, H_score, _, H_re, match_map = self.match_WH(W_est, H_est, W_ref, H_ref) 

            if prev_map != match_map:
                print("\033[93m" + 'map change' + "\033[0m")
            prev_map = match_map

            W_score = self.compare_W(W_ref, W_est, match_map)

            W_scores.append(W_score)
            H_scores.append(H_score)
            
            # reorder correct match
            print('iter', i, 'H_est', W_score)
            print_mat(H_est)
            print(match_map)
            print()

        # W_score, H_score, _, H_re, match_map = self.match_WH(W_est, H_est, W_ref, H_ref) 
        # print('print', W_score, H_score)
        # print_mat(H_re)

        iters = [i for i in range(max_iter)]
        fig, axs = plt.subplots(2)
        axs[0].scatter(iters, W_scores)
        axs[0].set_title(self.filepath)
        axs[1].scatter(iters, H_scores)

        plt.show()
        fig.savefig(self.filepath)

    def single_mf(self, X_input, prev_W, prev_H, init_new):
        W_init, H_init = None, None
        if init_new:
            W_init, H_init = nndsvd.initial_nndsvd(X_input, self.L, 10)  
        else:
            W_init, H_init = prev_W, prev_H

        W_est, H_est, opt = solver.alternate_minimize(
                W_init, H_init, X_input, 
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
        best = 999
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

        

